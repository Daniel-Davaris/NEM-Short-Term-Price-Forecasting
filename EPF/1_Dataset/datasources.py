from pathlib import Path
import logging
import math
import shutil
import subprocess
import textwrap
import traceback
import zipfile
import urllib.request
import urllib.error
import nemosis
import pandas as pd
import io
import openpyxl
import pyarrow as pa


logging.getLogger("nemosis").setLevel(logging.WARNING)


"""
A generic scheduler function to fetch datasource data by months
and skip re pulling existing months
"""

def _month_compiler(
        datasource_fetch_function: callable,
        start_date:pd.Timestamp,
        end_date:pd.Timestamp,
        datasource_file_path:Path,
        fallback_fetch_function: callable = None,
        cache_dir: Path = Path("Pre_processing/temporary_cache")
        ):

    cache_dir = Path(cache_dir)

    # Each processed month is streamed to its own part file under this directory
    # rather than re-loading and rewriting the entire cumulative parquet every
    # month (which held ~3 copies of the whole dataset in RAM and triggered OOM
    # kills). The single output parquet is assembled once, at the end, from these
    # parts.
    parts_dir = datasource_file_path.parent / f"{datasource_file_path.stem}_parts"

    datasource_file_year_month_range = set()
    # Check if the datasource file already exists, if so, check which years/months it has
    if datasource_file_path.exists() and datasource_file_path.stat().st_size > 0:
        try:
            # Extract only the SETTLEMENTDATE index to check which months are already present
            datasource_file_date_range = pd.read_parquet(datasource_file_path, columns=[]).index
            # Rows already stored per calendar month. A month is only treated as
            # done if it holds close to a full month of data — a month left with
            # just a stray boundary/carryover row (from a failed download that
            # returned near-nothing) is far below the typical month size and gets
            # re-fetched instead of being skipped as "processed" forever.
            month_counts = datasource_file_date_range.to_period("M").value_counts()
            complete_threshold = max(2, 0.5 * month_counts.median()) if len(month_counts) else 0
            row_ok = set(month_counts[month_counts >= complete_threshold].index)

            # Row-count alone is not enough for the forecast sources (predispatch /
            # pdpasa): _clean_up reindexes every month onto the full 5-min grid, so a
            # month that was downloaded before its data actually existed is stored
            # with a full row-count but all-NaN values. Detect those from a bounded
            # sample of value columns and re-fetch them too. The 90% cell-NaN cutoff
            # sits well above legitimately-sparse per-DUID bid months (<=57% NaN) and
            # far-horizon pdpasa sparsity (~25%), so only genuine empty months trip it.
            value_ok = set(row_ok)
            try:
                import pyarrow.parquet as _pq
                _cols = [c for c in _pq.ParquetFile(datasource_file_path).schema.names
                         if c != (datasource_file_date_range.name or "index")]
                if _cols:
                    _step = max(1, len(_cols) // 100)
                    _sample = pd.read_parquet(datasource_file_path, columns=_cols[::_step][:100])
                    _nan_frac = _sample.isna().groupby(_sample.index.to_period("M")).mean().mean(axis=1)
                    value_ok = {m for m in row_ok if _nan_frac.get(m, 0.0) < 0.90}
            except Exception as e:
                print(f"  (value-NaN check skipped: {e})", flush=True)

            datasource_file_year_month_range = value_ok
            req_lo = start_date.to_period("M")
            req_hi = (end_date - pd.Timedelta(days=1)).to_period("M")
            incomplete = sorted(
                str(m) for m in month_counts.index
                if m not in datasource_file_year_month_range and req_lo <= m <= req_hi
            )
            print(f"  Found {len(datasource_file_year_month_range)} complete month(s) — will skip.", flush=True)
            if incomplete:
                print(f"  {len(incomplete)} incomplete month(s) will be re-fetched: {incomplete}", flush=True)
        except Exception as e:
            # A corrupted / partially-written parquet (e.g. from an interrupted run)
            # would otherwise crash the whole task. Discard it and rebuild from scratch.
            print(f"  Existing file unreadable ({e}) — discarding and rebuilding from scratch.", flush=True)
            datasource_file_path.unlink()
            datasource_file_year_month_range = set()

    # Also skip months already streamed to a part file by an interrupted run.
    if parts_dir.exists():
        for part in parts_dir.glob("*.parquet"):
            try:
                datasource_file_year_month_range.add(pd.Period(part.stem, freq="M"))
            except Exception:
                pass

    # Set a var for total required data range
    datasource_required_total_range = pd.date_range(start_date, end_date - pd.Timedelta(days=1), freq="MS")

    # Loop through the datasource_required_total_range
    for i, month in enumerate(datasource_required_total_range):
        # Month in datasource_file_year_month_range so skip iteration
        if month.to_period("M") in datasource_file_year_month_range:
            print(f"{i + 1:3d}/{len(datasource_required_total_range)} {month:%Y-%m} skipping (already processed).", flush=True)
            continue
        # Month NOT in datasource_file_year_month_range so call datasource fetch function
        else:
            # Define the start of the calander month  
            current_month_start = month.strftime("%Y/%m/%d %H:%M:%S")
            # Define the end of the calander month
            current_month_end = (month + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)).strftime("%Y/%m/%d %H:%M:%S")
            
            print(f"{i + 1:3d}/{len(datasource_required_total_range)} {month:%Y-%m} fetching...", flush=True)
            
            # Call the datasource fetch function for the current month
            try:
                datasource_data = datasource_fetch_function(current_month_start, current_month_end)
            except nemosis.custom_errors.NoDataToReturn:
                # nemosis couldn't find the file it expects for this month (e.g. AEMO
                # renamed/retired the table in the archive). Try the fallback fetch
                # before giving up, and only skip if that fails too.
                if fallback_fetch_function is not None:
                    print(f"  {month:%Y-%m} nemosis found no data — trying archive fallback...", flush=True)
                    try:
                        datasource_data = fallback_fetch_function(current_month_start, current_month_end)
                        print(f"  {month:%Y-%m} fallback succeeded.", flush=True)
                    except Exception as backup_err:
                        print(f"  {month:%Y-%m} SKIPPED — fallback also failed: {backup_err}", flush=True)
                        continue
                else:
                    print(f"  {month:%Y-%m} SKIPPED — no data available from AEMO (nemosis NoDataToReturn).", flush=True)
                    continue
            except RuntimeError as e:
                if "404" in str(e) or "not available from MMS Historical Data SQL Loader" in str(e):
                    if fallback_fetch_function is not None:
                        print(f"  {month:%Y-%m} primary 404 — falling back to per-run archive...", flush=True)
                        try:
                            datasource_data = fallback_fetch_function(current_month_start, current_month_end)
                            print(f"  {month:%Y-%m} backup succeeded.", flush=True)
                        except Exception as backup_err:
                            print(f"  {month:%Y-%m} SKIPPED — backup also failed: {backup_err}", flush=True)
                            continue
                    else:
                        print(f"  {month:%Y-%m} SKIPPED — 404 Not Found (file unavailable on AEMO server).", flush=True)
                        continue
                else:
                    raise

            # nemosis may only warn and return an empty frame or a single
            # boundary/carryover row (instead of raising) when it can't download a
            # month's file, so a result with no rows inside the requested month is
            # also treated as a miss and routed through the fallback.
            def _rows_in_month(data):
                if data is None or data.empty:
                    return 0
                try:
                    idx = pd.to_datetime(data.index)
                    return int(((idx >= pd.Timestamp(current_month_start)) &
                                (idx < pd.Timestamp(current_month_end))).sum())
                except Exception:
                    return len(data)

            if _rows_in_month(datasource_data) == 0 and fallback_fetch_function is not None:
                print(f"  {month:%Y-%m} primary returned no rows for this month — trying archive fallback...", flush=True)
                try:
                    datasource_data = fallback_fetch_function(current_month_start, current_month_end)
                    print(f"  {month:%Y-%m} fallback succeeded.", flush=True)
                except Exception as backup_err:
                    print(f"  {month:%Y-%m} SKIPPED — fallback also failed: {backup_err}", flush=True)
                    continue

            # Never save a part with no rows for this month — it would mask the gap.
            if _rows_in_month(datasource_data) == 0:
                print(f"  {month:%Y-%m} SKIPPED — no rows returned.", flush=True)
                continue

            # Ensure the index name is set correctly
            datasource_data.index.name = "Date"
            # Write only this month to its own part file — no full-dataset reload.
            parts_dir.mkdir(parents=True, exist_ok=True)
            datasource_data.to_parquet(parts_dir / f"{month:%Y-%m}.parquet")
            del datasource_data

            # Empty the temporary cache (Deletes all files and subdirectories)
            if cache_dir.exists():
                for entry in cache_dir.iterdir():
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()

            print(f"  {month:%Y-%m} saved & raw files deleted.", flush=True)

    # Assemble the single output parquet once, from the per-month parts (plus any
    # previously completed file), then drop the parts directory. The output is
    # streamed one month at a time to a ParquetWriter rather than built from a
    # single pd.concat(...).sort_index(): a wide forecast table (predispatch
    # REGIONSUM is ~1950 columns → ~14 GB for one in-memory copy) needs 2-3 whole
    # copies for concat+dedup+sort, which OOM-killed the kernel. Streaming holds at
    # most one copy of the existing file plus one month. Re-fetched month parts
    # override the stored month of the same name.
    if parts_dir.exists() and any(parts_dir.glob("*.parquet")):
        import gc
        import os as _os
        import pyarrow.parquet as _pq

        part_files = {p.stem: p for p in parts_dir.glob("*.parquet")}  # "YYYY-MM" -> Path
        have_existing = datasource_file_path.exists() and datasource_file_path.stat().st_size > 0

        existing = pd.read_parquet(datasource_file_path) if have_existing else None
        existing_periods = None
        if existing is not None:
            existing = existing[~existing.index.duplicated(keep="last")]
            existing_periods = existing.index.to_period("M")

        # Fixed master column set = union of existing + every part, so each month is
        # written with an identical schema (missing entity/horizon combos become
        # NaN) — reproduces the column-union that pd.concat performed implicitly.
        master_cols = list(existing.columns) if existing is not None else []
        seen = set(master_cols)
        for p in part_files.values():
            for c in _pq.ParquetFile(p).schema_arrow.names:
                if c not in seen and c != "Date":
                    seen.add(c)
                    master_cols.append(c)

        months = set(part_files)
        if existing_periods is not None:
            months |= {str(x) for x in existing_periods.unique()}

        tmp_path = datasource_file_path.with_suffix(".assembling.parquet")
        writer = None
        try:
            for m in sorted(months):  # "YYYY-MM" sorts chronologically
                if m in part_files:
                    dfm = pd.read_parquet(part_files[m])
                    dfm = dfm[~dfm.index.duplicated(keep="last")]
                elif existing is not None:
                    dfm = existing.loc[existing_periods == pd.Period(m, freq="M")]
                else:
                    continue
                if dfm.empty:
                    continue
                dfm = dfm.sort_index().reindex(columns=master_cols)
                dfm.index.name = "Date"
                table = pa.Table.from_pandas(dfm, preserve_index=True)
                if writer is None:
                    writer = _pq.ParquetWriter(tmp_path, table.schema)
                writer.write_table(table)
                del dfm, table
                gc.collect()
        finally:
            if writer is not None:
                writer.close()

        del existing
        gc.collect()
        _os.replace(tmp_path, datasource_file_path)
        shutil.rmtree(parts_dir)

    # Return the path, not the data: callers discard the value, so loading the
    # full (multi-GB) dataset here would only add avoidable memory pressure.
    return datasource_file_path


"""
A generic function to fetch datasource data in one shot
"""

def _one_shot_compiler(
        datasource_fetch_function: callable,
        start_date:pd.Timestamp,
        end_date:pd.Timestamp,
        datasource_file_path:Path
        ):
    
    
        datasource_data = datasource_fetch_function(start_date, end_date)
        datasource_data.index.name = "SETTLEMENTDATE"
        datasource_data.to_parquet(datasource_file_path)

        return pd.read_parquet(datasource_file_path)


"""
Datasource 1
"""
def _dispatch_price(start: str, end: str, cache_dir="Pre_processing/temporary_cache"):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    region_columns = {"NSW1": "nsw_price", "QLD1": "qld_price", "VIC1": "vic_price", "SA1": "sa_price"}

    def _API_call():

        API_response = nemosis.dynamic_data_compiler(
            start_time=start,
            end_time=end,
            table_name="DISPATCHPRICE",
            raw_data_location=str(cache_dir),
            select_columns=["SETTLEMENTDATE", "REGIONID", "RRP"],
            filter_cols=["REGIONID"],
            filter_values=[list(region_columns.keys())],
            fformat="feather",
            keep_csv=False,
        )
        
        return API_response

    def _clean_up(API_response):
        # Handle conversion of data types
        API_response["SETTLEMENTDATE"] = pd.to_datetime(API_response["SETTLEMENTDATE"])
        API_response["RRP"] = pd.to_numeric(API_response["RRP"], errors="coerce")

        # Handle data granularity
        API_response = (
            API_response.groupby(["SETTLEMENTDATE", "REGIONID"])["RRP"]
            .mean()
            .unstack("REGIONID")
            .resample("5min").mean()
            .asfreq("5min")
        )

        # Rename columns
        API_response = API_response.rename(columns=region_columns)

        return API_response

    data = _API_call()
    data_clean = _clean_up(data)
    data_clean.index.name = "Date"
    return data_clean


"""
Datasource 2
"""
def _dispatch_region_sum(start: str, end: str, cache_dir="Pre_processing/temporary_cache"):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    RAW_FIELDS    = ["TOTALDEMAND", "AVAILABLEGENERATION", "NETINTERCHANGE", "DEMANDFORECAST", "DISPATCHABLEGENERATION"]
    REGION_SUFFIX = {"NSW1": "_nsw", "QLD1": "_qld", "VIC1": "_vic", "SA1": "_sa"}

    def _API_call():
        
        API_response = nemosis.dynamic_data_compiler(
            start_time=start,
            end_time=end,
            table_name="DISPATCHREGIONSUM",
            raw_data_location=str(cache_dir),
            select_columns=["SETTLEMENTDATE", "REGIONID"] + RAW_FIELDS,
            filter_cols=["REGIONID"],
            filter_values=[list(REGION_SUFFIX.keys())],
            fformat="feather",
            keep_csv=False,
        )
        
        return API_response

    def _clean_up(API_response):

        # Handle conversion of data types
        OUTPUT_FIELDS = ["demand", "avail_gen", "interchange", "demand_forecast", "dispatch_gen"]

        API_response["SETTLEMENTDATE"] = pd.to_datetime(API_response["SETTLEMENTDATE"])

        for col in RAW_FIELDS:
            API_response[col] = pd.to_numeric(API_response[col], errors="coerce")

        # Handle data granularity / transform from long format
        region_frames = []
        for region, suffix in REGION_SUFFIX.items():
            rdf = API_response[API_response["REGIONID"] == region][["SETTLEMENTDATE"] + RAW_FIELDS].copy()
            rdf = (
                rdf.set_index("SETTLEMENTDATE")
                .sort_index()
                .resample("5min")
                .mean(numeric_only=True)
            )
            rdf.columns = [f"{name}{suffix}" for name in OUTPUT_FIELDS]
            region_frames.append(rdf)
            
        API_response = pd.concat(region_frames, axis=1).sort_index()

        return API_response

    data = _API_call()
    data_clean = _clean_up(data)
    data_clean.index.name = "Date"
    return data_clean


def _nem_registration_and_exemption_list():

    def _API_call():
        cache_path = Path("Pre_processing/temporary_cache") / "NEM Registration and Exemption List.xlsx"
        url = "https://www.aemo.com.au/-/media/files/electricity/nem/participant_information/nem-registration-and-exemption-list.xlsx"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(cache_path, "wb") as f:
            f.write(resp.read())
        try:
            df = pd.read_excel(cache_path, sheet_name="PU and Scheduled Loads", engine="openpyxl", header=0)
        finally:
            cache_path.unlink(missing_ok=True)
        return df
            
    def _clean_up(df):
        df = df[df["Dispatch Type"].astype(str).str.upper() != "LOAD"]
        df = df[df["DUID"].astype(str).str.upper() != "-"].copy()
        df["DUID"] = df["DUID"].astype(str).str.strip()
        df["Fuel Source - Primary"] = df["Fuel Source - Primary"].fillna("").astype(str).str.strip()
        df = df[df["Fuel Source - Primary"] != ""].copy()
        df = df[df["Category"].astype(str).str.strip().str.upper() == "MARKET"].copy()
        return df

    def _assign_fuel_type(df):
        mask = df["Fuel Source - Primary"].str.lower().str.contains("biomass", na=False)
        df.loc[mask, "Fuel Source - Primary"] = "Biomass"

        fossil_mask = df["Fuel Source - Primary"].astype(str).str.strip().str.lower().eq("fossil")
        coal_mask = df["Fuel Source - Descriptor"].astype(str).str.strip().isin({"Black Coal", "Brown Coal", "Coal Seam Methane"})

        df.loc[fossil_mask & coal_mask, "Fuel Source - Primary"] = "Coal"
        df.loc[fossil_mask & ~coal_mask, "Fuel Source - Primary"] = "Gas"

        df["Region"] = (df["Region"].astype(str).str.strip().str.replace(r"1$", "", regex=True))

        return df[["Region", "Participant", "Station Name", "DUID","Classification","Fuel Source - Primary"]].reset_index(drop=True)
       
    df = _API_call()
    df = _clean_up(df)
    df = _assign_fuel_type(df)
    df.to_parquet("Processed_data/0_nem_duid_mapping.parquet")
    return df


"""
Datasource 3
"""
def _generation_fuel(start: str, end: str, cache_dir="Pre_processing/temporary_cache"):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _API_call(duid_to_col):
        API_response = nemosis.dynamic_data_compiler(
            start_time=start,
            end_time=end,
            table_name="DISPATCH_UNIT_SCADA",
            raw_data_location=str(cache_dir),
            select_columns=["SETTLEMENTDATE", "DUID", "SCADAVALUE"],
            filter_cols=["DUID"],
            filter_values=[sorted(duid_to_col.keys())],
            fformat="feather",
            keep_csv=False,
        )
        return API_response

    def _clean_up(API_response, duid_to_col):
        battery_duids = {duid for duid, col in duid_to_col.items() if col.startswith("battery_mw_")}

        API_response["SETTLEMENTDATE"] = pd.to_datetime(API_response["SETTLEMENTDATE"])
        API_response["SCADAVALUE"]     = pd.to_numeric(API_response["SCADAVALUE"], errors="coerce")
        API_response["_col"]           = API_response["DUID"].map(duid_to_col)
        API_response = API_response[API_response["_col"].notna()].copy()

        API_response["_target"] = API_response["_col"]
        API_response["_value"]  = API_response["SCADAVALUE"].clip(lower=0)
        is_battery = API_response["DUID"].isin(battery_duids)
        if is_battery.any():
            charge    = is_battery & (API_response["SCADAVALUE"] < 0)
            discharge = is_battery & (API_response["SCADAVALUE"] >= 0)
            API_response.loc[charge,    "_target"] = API_response.loc[charge,    "_col"].str.replace("battery_mw_", "battery_charge_mw_",    regex=False)
            API_response.loc[charge,    "_value"]  = -API_response.loc[charge,   "SCADAVALUE"]
            API_response.loc[discharge, "_target"] = API_response.loc[discharge, "_col"].str.replace("battery_mw_", "battery_discharge_mw_", regex=False)
            API_response.loc[discharge, "_value"]  = API_response.loc[discharge, "SCADAVALUE"].clip(lower=0)

        API_response = (
            API_response.groupby(["SETTLEMENTDATE", "_target"])["_value"]
            .sum()
            .unstack("_target")
            .resample("5min").mean()
            .fillna(0)
        )

        # Ensure every fuel×region combination is present, even if always 0.
        all_regions  = sorted({col.split("_mw_")[1] for col in duid_to_col.values()})
        all_fuels    = sorted({col.split("_mw_")[0] for col in duid_to_col.values() if not col.startswith("battery_mw_")})
        has_battery  = any(col.startswith("battery_mw_") for col in duid_to_col.values())
        non_battery  = [f"{fuel}_mw_{region}" for fuel in all_fuels for region in all_regions]
        battery_cols = [f"battery_{d}_mw_{region}" for d in ("charge", "discharge") for region in all_regions] if has_battery else []
        all_expected = sorted(set(non_battery + battery_cols), key=lambda c: (c.split("_mw_")[1], c.split("_mw_")[0]))
        API_response = API_response.reindex(columns=all_expected, fill_value=0.0)

        return API_response

    duid_map = pd.read_parquet("Processed_data/0_nem_duid_mapping.parquet")
    duid_map = duid_map[~duid_map["Region"].str.upper().isin({"TAS"})]
    def _fuel_key(fuel: str) -> str:
        f = fuel.lower().strip()
        return "battery" if "battery" in f else f
    duid_to_col = {
        row["DUID"]: f"{_fuel_key(row['Fuel Source - Primary'])}_mw_{row['Region'].lower()}"
        for _, row in duid_map.iterrows()
    }
    data = _API_call(duid_to_col)
    data_clean = _clean_up(data, duid_to_col)
    data_clean.index.name = "Date"
    return data_clean


"""
Datasource 4
"""

def _STTM_DWGM(start: str, end: str):
    

    def _load_1():
        request = urllib.request.Request(
            "https://www.aemo.com.au/-/media/files/gas/sttm/data/sttm-price-and-withdrawals.xlsx",
            headers={"User-Agent": "Mozilla/5.0"},
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            data = openpyxl.load_workbook(io.BytesIO(response.read()), read_only=True, data_only=True)

        return data
    

    def _load_2():
        request = urllib.request.Request(
            "https://www.aemo.com.au/-/media/files/gas/dwgm/dwgm-prices-and-demand.xlsx",
            headers={"User-Agent": "Mozilla/5.0"},
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            data = openpyxl.load_workbook(io.BytesIO(response.read()), read_only=True, data_only=True)

        return data
    

    def _process(data1, data2):

        idx = pd.date_range(start=start, end=end, freq="5min", name="Date")

        sheets = {
            "SYD price and withdrawals": "gas_price_nsw",
            "ADL price and withdrawals": "gas_price_sa",
            "BRI price and withdrawals": "gas_price_qld",
        }

        data_clean = {}
        for sheet_name, col_name in sheets.items():
            rows = [(pd.Timestamp(r[0]), float(r[1])) for r in data1[sheet_name].iter_rows(min_row=2, values_only=True) if r[0] is not None and r[1] is not None]
            daily = pd.Series(dict(rows))
            data_clean[col_name] = daily.reindex(idx.normalize()).set_axis(idx).ffill().bfill()

        # DWGM (VIC) — 5 sub-daily intervals, ffill within each ~4-hour horizon
        rows_vic = [(pd.Timestamp(r[0]) + pd.Timedelta(hours=int(r[1])), float(r[2])) for r in data2["Prices"].iter_rows(min_row=2, values_only=True) if r[0] is not None and r[1] is not None and r[2] is not None]
        dwgm = pd.Series(dict(rows_vic), name="gas_price_vic").sort_index()
        data_clean["gas_price_vic"] = dwgm.reindex(idx, method="ffill").bfill()

        return pd.DataFrame(data_clean, index=idx)
    


    data1, data2 = _load_1(), _load_2()
    data_clean = _process(data1,data2)
    return data_clean


"""
Datasource 5
"""

def _weather(start: str, end: str):
    
    def _load():
        sydney    = pd.read_csv("Pre_processing/weather_data/NSW weather.csv")
        brisbane  = pd.read_csv("Pre_processing/weather_data/QLD weather.csv")
        melbourne = pd.read_csv("Pre_processing/weather_data/VIC weather.csv")
        adelaide  = pd.read_csv("Pre_processing/weather_data/SA weather.csv")

        return sydney, brisbane, melbourne, adelaide

    def _process(data: pd.DataFrame, city: str) -> pd.DataFrame:
        data["datetime"] = pd.to_datetime(data["datetime"])
        data = data.set_index("datetime").sort_index()
        data = data[~data.index.duplicated(keep="first")]
        data = data.rename(columns={col: f"{str(col).strip().lower().replace(' ', '')}_{city}" for col in data.columns})
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.dropna(axis=1, how="all")
        return data.resample(f"{5}min").interpolate(method="time")
    
    sydney, brisbane, melbourne, adelaide = _load()

    sydney    = _process(sydney, "sydney")
    brisbane  = _process(brisbane, "brisbane")
    melbourne = _process(melbourne, "melbourne")
    adelaide  = _process(adelaide,  "adelaide")

   
    data_clean = pd.concat([sydney, brisbane, melbourne, adelaide], axis=1)
    data_clean.index.name = "Date"
   
    return data_clean


def _nemseer_pull(
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        datasource_file_path: Path,
        nemseer_forecast_type: str,
        nemseer_table_name: str,
        value_cols: list,
        run_col: str,
        interval_col: str,
        entity_col: str,
        cache_dir: Path = Path("Pre_processing/temporary_cache"),
        ):

    cache_dir = Path(cache_dir)

    def _repeat_logic(start: str, end: str, _use_backup: bool = False, _use_mmsdm: bool = False):

        modified_start = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y/%m/%d %H:%M")

        def _API_call_default():
            # nemseer call requires two separate temp folders
            (cache_dir / "raw").mkdir(parents=True, exist_ok=True)
            (cache_dir / "processed").mkdir(parents=True, exist_ok=True)

            def get_special_dates():
                # Compute max allowed forecasted_end: nemseer caps at next-trading-day 04:00, Trading days run 04:00–04:00 AEST
                run_end_ts = pd.Timestamp(end[:16])
                current_trading_day_end = run_end_ts.normalize() + pd.Timedelta(hours=4)
                if run_end_ts >= current_trading_day_end:
                    current_trading_day_end += pd.Timedelta(days=1)
                return current_trading_day_end.strftime("%Y/%m/%d %H:%M")

            forecasted_end = get_special_dates()

            subprocess_code = textwrap.dedent(f"""
                import logging
                logging.getLogger("nemseer").setLevel(logging.WARNING)
                import pandas as pd
                pd.options.future.infer_string = False
                import nemseer, sys
                data = nemseer.compile_data(
                    run_start={modified_start!r},
                    run_end={end[:16]!r},
                    forecasted_start={modified_start!r},
                    forecasted_end={forecasted_end!r},
                    raw_cache={str(cache_dir / "raw")!r},
                    processed_cache={str(cache_dir / "processed")!r},
                    forecast_type={nemseer_forecast_type!r},
                    tables=[{nemseer_table_name!r}],
                )
                df = data[{nemseer_table_name!r}]
                if df is None or df.empty:
                    raise ValueError("nemseer returned no data")
                df.reset_index().to_csv(sys.stdout, index=False)
            """)
            
            path = "/home/ec2-user/venv_sub/bin/python"  # EC2 Linux: python3.11 venv

            result = subprocess.run(
                [path, "-c", subprocess_code],
                capture_output=True, text=True, cwd=Path.cwd(),
            )

            # Needs this to actually display errors from the subprocess
            if result.returncode != 0 or not result.stdout.strip():
                raise RuntimeError(f"nemseer subprocess failed:\n{result.stderr}")

            API_response = pd.read_csv(io.StringIO(result.stdout))
            return API_response

        def _API_call_backup():
            """
            Fallback: fetches pre-dispatch data from the NEMWeb weekly archive at
            nemweb.com.au/Reports/Archive/PreDispatch_Reports/.
            Files are named PUBLIC_PREDISPATCH_YYYYMMDD_YYYYMMDD.zip (one per week).
            Scrapes the directory listing, downloads weekly ZIPs that overlap
            [modified_start, end), parses AEMO's I/D CSV record format, and returns
            a DataFrame in the same shape as _API_call_default.
            Only PREDISPATCH forecast type is supported.
            Note: the archive retains roughly the last 12 months of data.

            Archive format notes (differs from nemseer):
              - Table identifier is in parts[1] (DataSource), not parts[2] (TableName)
              - Column names differ: PREDISPATCHSEQNO → run_col, PERIODID → interval_col
              - Values are quoted CSV, so csv.reader is used instead of str.split
            """
            import re
            import csv as csv_mod

            ARCHIVE_ROOTS = {
                "PREDISPATCH": "https://nemweb.com.au/Reports/ARCHIVE/Predispatch_Reports/",
            }
            if nemseer_forecast_type not in ARCHIVE_ROOTS:
                raise RuntimeError(
                    f"Per-run backup not supported for forecast type '{nemseer_forecast_type}'. "
                    "Only PREDISPATCH is implemented."
                )

            root         = ARCHIVE_ROOTS[nemseer_forecast_type]
            window_start = pd.Timestamp(modified_start[:16])
            window_end   = pd.Timestamp(end[:16])

            # Scrape directory listing to find weekly ZIPs that overlap the window
            req = urllib.request.Request(root, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            zip_urls = []
            for href in re.findall(r'href=["\']([^"\']*\.zip)["\']', html, re.IGNORECASE):
                fname = href.rsplit("/", 1)[-1]
                # Filename: PUBLIC_PREDISPATCH_YYYYMMDD_YYYYMMDD.zip
                m = re.search(r"PUBLIC_PREDISPATCH_(\d{8})_(\d{8})\.zip", fname, re.IGNORECASE)
                if not m:
                    continue
                try:
                    zip_start = pd.to_datetime(m.group(1), format="%Y%m%d")
                    zip_end   = pd.to_datetime(m.group(2), format="%Y%m%d") + pd.Timedelta(days=1)
                except Exception:
                    continue
                # Include the ZIP if its date range overlaps with [window_start, window_end)
                if zip_start < window_end and zip_end > window_start:
                    full_url = href if href.startswith("http") else root + fname
                    zip_urls.append((zip_start, full_url))

            if not zip_urls:
                raise RuntimeError(
                    f"Backup: no weekly ZIP files found overlapping {window_start} to "
                    f"{window_end} at {root}. The archive only retains ~12 months of data."
                )

            print(f"    Backup archive: found {len(zip_urls)} weekly ZIP(s) — downloading and parsing...", flush=True)

            # AEMO archive uses different table identifiers (in parts[1]) and column names
            # than nemseer. Map nemseer table name → (archive_table_id, {archive_col: nemseer_col})
            PREDISPATCH_TABLE_MAP = {
                "PRICE":             ("PDREGION",       {"PREDISPATCHSEQNO": run_col, "PERIODID": interval_col}),
                "REGIONSUM":         ("PDREGION",       {"PREDISPATCHSEQNO": run_col, "PERIODID": interval_col}),
                "INTERCONNECTORRES": ("PDINTERCONNECT", {"PREDISPATCHSEQNO": run_col, "PERIODID": interval_col}),
            }
            table_entry      = PREDISPATCH_TABLE_MAP.get(nemseer_table_name.upper())
            archive_table_id = table_entry[0] if table_entry else nemseer_table_name.upper()
            archive_col_map  = table_entry[1] if table_entry else {}  # archive_col → nemseer_col

            # Build archive_want: archive_col_upper → nemseer_col_name (used when extracting D rows)
            nemseer_cols       = [run_col, interval_col, entity_col] + value_cols
            nemseer_to_archive = {v: k for k, v in archive_col_map.items()}
            archive_want       = {nemseer_to_archive.get(c, c).upper(): c for c in nemseer_cols}

            all_rows          = []
            seen_tables       = set()
            first_csv_preview = []

            def _parse_text(text):
                header      = None
                col_indices = {}
                for row in csv_mod.reader(text.splitlines()):
                    if not row:
                        continue
                    rec = row[0].upper()
                    if rec == "I":
                        # Track both identifier positions for diagnostics
                        if len(row) > 1: seen_tables.add(row[1].upper())
                        if len(row) > 2: seen_tables.add(row[2].upper())
                    if len(row) < 5:
                        continue
                    # Archive stores table ID in parts[1] (DataSource), not parts[2] (TableName)
                    table_here = (
                        row[1].upper() == archive_table_id or
                        row[2].upper() == archive_table_id
                    )
                    if rec == "I" and table_here:
                        header      = [c.upper() for c in row[4:]]
                        col_indices = {col: i for i, col in enumerate(header)}
                    elif rec == "D" and col_indices and table_here:
                        vals = row[4:]
                        r    = {
                            orig: vals[col_indices[up]]
                            for up, orig in archive_want.items()
                            if up in col_indices and col_indices[up] < len(vals)
                        }
                        if len(r) == len(archive_want):
                            all_rows.append(r)

            def _extract_recursive(zf, depth=0):
                """
                Recursively parse CSVs from a ZIP, handling nested ZIPs up to depth 3.
                AEMO archive structure can be:
                  weekly ZIP → per-run ZIPs → CSVs  (2 levels), or
                  weekly ZIP → daily ZIPs → per-run ZIPs → CSVs  (3 levels).
                """
                for name in zf.namelist():
                    if name.upper().endswith(".CSV"):
                        try:
                            with zf.open(name) as f:
                                text = f.read().decode("utf-8", errors="replace")
                            # Capture the first 8 lines of the very first CSV seen for diagnostics
                            if not first_csv_preview:
                                first_csv_preview.extend(text.splitlines()[:8])
                            _parse_text(text)
                        except Exception as e:
                            print(f"      Warning: could not parse CSV {name}: {e}", flush=True)
                    elif name.upper().endswith(".ZIP") and depth < 3:
                        try:
                            with zf.open(name) as inner_bytes:
                                with zipfile.ZipFile(io.BytesIO(inner_bytes.read())) as inner_zf:
                                    _extract_recursive(inner_zf, depth + 1)
                        except Exception as e:
                            print(f"      Warning: could not open nested ZIP {name}: {e}", flush=True)

            for file_num, (_, url) in enumerate(sorted(zip_urls), 1):
                print(f"    Backup archive: downloading weekly ZIP {file_num}/{len(zip_urls)}...", flush=True)
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        zdata = resp.read()
                    with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
                        _extract_recursive(zf)
                except Exception as e:
                    print(f"      Warning: failed to download/process {url.rsplit('/', 1)[-1]}: {e}", flush=True)

            if not all_rows:
                preview_str = "\n".join(first_csv_preview) if first_csv_preview else "(no CSVs reached)"
                raise RuntimeError(
                    f"Backup archive: no rows parsed for table '{nemseer_table_name}' "
                    f"(archive ID: '{archive_table_id}') between {window_start} and {window_end}.\n"
                    f"  Tables found in I-records: {sorted(t for t in seen_tables if t)}\n"
                    f"  Archive want columns: {sorted(archive_want)}\n"
                    f"  First CSV preview:\n{preview_str}"
                )

            return pd.DataFrame(all_rows)

        def _API_call_mmsdm():
            """
            Second fallback: AEMO MMSDM monthly historical archive at
            nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/.
            Covers all historical months. Supports PREDISPATCH (PRICE, REGIONSUM,
            INTERCONNECTORRES) and PDPASA (REGIONSOLUTION).

            PREDISPATCH files: LASTCHANGED floored to 30-min is used as the run
            timestamp (no dedicated run col in MMSDM); DATETIME is the interval col.
            PDPASA files: RUN_DATETIME and INTERVAL_DATETIME are directly available.
            """
            import re as _re, csv as _csv

            # Maps (forecast_type, table_name) → MMSDM file search keyword (matched against HREF)
            MMSDM_FILE_KEY = {
                ("PREDISPATCH", "PRICE"):             "PREDISPATCHPRICE",
                ("PREDISPATCH", "REGIONSUM"):         "PREDISPATCHREGIONSUM",
                ("PREDISPATCH", "INTERCONNECTORRES"): "PREDISPATCHINTERCONNECTORRES",
                ("PDPASA",      "REGIONSOLUTION"):    "PDPASA_REGIONSOLUTION",
            }
            # row[1] value in MMSDM D/I records for each forecast type
            MMSDM_SOURCE_ID = {
                "PREDISPATCH": "PREDISPATCH",
                "PDPASA":      "PDPASA",
            }
            # MMSDM column to use as run timestamp, and whether to floor it to 30-min
            # PREDISPATCH has no dedicated run col; LASTCHANGED is the closest proxy
            MMSDM_RUN_COL = {
                "PREDISPATCH": ("LASTCHANGED",    True),   # (col_name, floor_to_30min)
                "PDPASA":      ("RUN_DATETIME",   False),
            }
            # MMSDM column name for the forecast interval timestamp
            MMSDM_INTERVAL_COL = {
                "PREDISPATCH": "DATETIME",
                "PDPASA":      "INTERVAL_DATETIME",
            }

            key        = (nemseer_forecast_type.upper(), nemseer_table_name.upper())
            search_key = MMSDM_FILE_KEY.get(key)
            if search_key is None:
                raise RuntimeError(
                    f"MMSDM backup not supported for ({nemseer_forecast_type}, {nemseer_table_name}). "
                    f"Supported combinations: {list(MMSDM_FILE_KEY.keys())}"
                )

            mmsdm_source_id          = MMSDM_SOURCE_ID[nemseer_forecast_type.upper()]
            mmsdm_run_col_name, floor_run = MMSDM_RUN_COL[nemseer_forecast_type.upper()]
            mmsdm_interval_col_name  = MMSDM_INTERVAL_COL[nemseer_forecast_type.upper()]

            month_ts = pd.Timestamp(start[:16]).replace(day=1)
            year, month = month_ts.year, month_ts.month

            # PREDISPATCH: the DATA/ folder only carries the reduced current-price
            # archive (PUBLIC_ARCHIVE#<TABLE>#FILE01, PERIODID=1 only). The full
            # multi-horizon data lives in PREDISP_ALL_DATA/, named either
            #   PUBLIC_ARCHIVE#<TABLE>#ALL#FILE0N   (newer months, ~79 periods), or
            #   PUBLIC_DVD_<TABLE>[part]_<yyyymm>   (older months, full DVD dump),
            # sometimes split across numbered parts. The regex accepts both and
            # excludes the reduced daily "<TABLE>_D_" and "<TABLE>SENSITIVIT*" files.
            # PDPASA keeps the single DATA/ file.
            if nemseer_forecast_type.upper() == "PREDISPATCH":
                data_folder = "PREDISP_ALL_DATA"
                match_pat   = _re.compile(
                    rf"{search_key.upper()}(%23ALL%23|[0-9]*_[0-9])", _re.IGNORECASE
                )
            else:
                data_folder = "DATA"
                match_pat   = _re.compile(_re.escape(search_key.upper()), _re.IGNORECASE)

            base_url = (
                f"https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"
                f"{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/{data_folder}/"
            )
            req = urllib.request.Request(base_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            links       = _re.findall(r'HREF="(/[^"]+)"', html)
            match_links = [
                l for l in links
                if l.upper().endswith(".ZIP") and match_pat.search(l.upper())
            ]
            if not match_links:
                avail = [
                    l.rsplit("%23", 2)[-2] if "%23" in l else l.rsplit("/", 1)[-1]
                    for l in links
                    if any(k in l.upper() for k in ("PREDISPATCH", "PDPASA"))
                    and l.upper().endswith(".ZIP")
                ]
                raise RuntimeError(
                    f"MMSDM archive: no file for '{search_key}' in {year}-{month:02d}. "
                    f"Available: {avail}"
                )

            # Build column want map: uppercase MMSDM col name → output col name
            want = {
                mmsdm_run_col_name.upper():      run_col,
                mmsdm_interval_col_name.upper(): interval_col,
                entity_col.upper():              entity_col,
            }
            for vc in value_cols:
                want[vc.upper()] = vc

            all_rows = []
            for mi, match_link in enumerate(sorted(match_links), 1):
                file_url = f"https://nemweb.com.au{match_link}"
                print(f"    MMSDM archive: downloading {year}-{month:02d} "
                      f"({search_key} part {mi}/{len(match_links)})...", flush=True)
                req = urllib.request.Request(file_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=300) as resp:
                    zdata = resp.read()

                with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
                    for name in zf.namelist():
                        if not name.upper().endswith(".CSV"):
                            continue
                        with zf.open(name) as f:
                            text = f.read().decode("utf-8", errors="replace")
                        col_idx          = {}
                        intervention_idx = None
                        for row in _csv.reader(text.splitlines()):
                            if not row or len(row) < 5:
                                continue
                            rec = row[0].upper()
                            if rec == "I" and row[1].upper() == mmsdm_source_id:
                                hdrs             = [c.upper() for c in row[4:]]
                                col_idx          = {c: i for i, c in enumerate(hdrs)}
                                intervention_idx = col_idx.get("INTERVENTION")
                            elif rec == "D" and col_idx and row[1].upper() == mmsdm_source_id:
                                vals = row[4:]
                                # Skip intervention runs where the column exists
                                if intervention_idx is not None and intervention_idx < len(vals):
                                    if vals[intervention_idx].strip() != "0":
                                        continue
                                r = {}
                                for up_col, out_col in want.items():
                                    if up_col in col_idx and col_idx[up_col] < len(vals):
                                        r[out_col] = vals[col_idx[up_col]]
                                if len(r) == len(want):
                                    all_rows.append(r)

            if not all_rows:
                raise RuntimeError(
                    f"MMSDM archive: no rows parsed for '{search_key}' in {year}-{month:02d}"
                )

            df = pd.DataFrame(all_rows)
            if floor_run:
                # Floor LASTCHANGED to nearest 30-min to approximate run boundaries
                df[run_col] = pd.to_datetime(df[run_col], format="mixed").dt.floor("30min")
            print(f"    MMSDM archive: parsed {len(df):,} rows.", flush=True)
            return df

        def _clean_up(API_response):

            # Handle conversion of data types
            API_response[run_col]      = pd.to_datetime(API_response[run_col], format="mixed")
            API_response[interval_col] = pd.to_datetime(API_response[interval_col], format="mixed")
            for col in value_cols:
                API_response[col] = pd.to_numeric(API_response[col], errors="coerce")

            # Build 5-min output grid indexed by delivery timestamp
            output_idx = pd.date_range(
                start[:16], end[:16],
                freq="5min", name="SETTLEMENTDATE"
            )
            # Extend T_df back to `start` (1 day before output window) so that runs
            # are included in the pivot — their h57+ values can then
            # be ffilled forward into the Jan 1 rows
            full_idx = pd.date_range(
                pd.Timestamp(modified_start[:16]), pd.Timestamp(end[:16]),
                freq="5min", name="SETTLEMENTDATE"
            )
            T_df = pd.DataFrame({"SETTLEMENTDATE": full_idx})

            # For each delivery timestamp T, find the most recently published run (run_time <= T)
            # This is the causally correct assignment: no future run information leaks into T
            run_times_df = (
                API_response[[run_col]]
                .drop_duplicates()
                .sort_values(run_col)
                .reset_index(drop=True)
            )
            T_df = pd.merge_asof(
                T_df, run_times_df,
                left_on="SETTLEMENTDATE", right_on=run_col,
                direction="backward"
            )

            # Join each T to all forecast rows from its assigned run
            merged = T_df.merge(API_response, on=run_col, how="left")

            # Drop T values where no prior run exists (no run available before T)
            merged = merged[merged[interval_col].notna()]

            # Compute horizon relative to delivery timestamp T using ceiling division:
            # h=1 → first 30-min period strictly after T, h=2 → second, etc.
            # ceil is required (not round) so that e.g. T=00:15 with interval=00:30
            # gives (900s / 1800) = 0.5 → ceil → 1, not round → 0 (which would drop it)
            merged["horizon"] = (
                (merged[interval_col] - merged["SETTLEMENTDATE"]).dt.total_seconds()
                .div(1800)
                .apply(math.ceil)
            )

            # Keep only forward-looking horizons. Cap at 78 periods (~39h ahead):
            # nemseer/weekly-archive runs stop there, but the MMSDM full archive
            # carries an extra period 79 — dropping it keeps one consistent column
            # set (h1..h78) across every source so re-fetched months align with the
            # months already stored.
            merged = merged[(merged["horizon"] >= 1) & (merged["horizon"] <= 78)]

            # Pivot each value column separately and concat
            frames = []
            for col in value_cols:
                pivot = (
                    merged.groupby(["SETTLEMENTDATE", entity_col, "horizon"])[col]
                    .mean()
                    .unstack([entity_col, "horizon"])
                )
                pivot = pivot.sort_index(axis=1)

                if entity_col == "REGIONID":
                    # Strip trailing digit: "NSW1" → "nsw"
                    pivot.columns = [
                        f"{nemseer_forecast_type.lower()}_{col.lower()}_{ent[:-1].lower()}_h{h}"
                        for ent, h in pivot.columns
                    ]
                else:
                    # Interconnector: replace dashes with underscores, e.g. "NSW1-QLD1" → "nsw1_qld1"
                    pivot.columns = [
                        f"{nemseer_forecast_type.lower()}_{col.lower()}_{ent.lower().replace('-', '_')}_h{h}"
                        for ent, h in pivot.columns
                    ]
                frames.append(pivot)

            result = pd.concat(frames, axis=1)
            result.index.name = "Date"

            # Reindex to the full 5-min grid, then ffill per horizon column:
            # carries the last known forecast for each horizon forward in time — no leakage
            # since ffill only looks backward (earlier timestamps)
            # full_idx covers the 3-day lookback so Dec 31 h57+ values ffill into Jan 1
            result = result.reindex(full_idx).ffill().reindex(output_idx)

            return result

        if _use_mmsdm:
            data = _API_call_mmsdm()
        elif _use_backup:
            data = _API_call_backup()
        else:
            data = _API_call_default()
        data_clean = _clean_up(data)
        return data_clean

    def _repeat_logic_backup(start: str, end: str):
        try:
            return _repeat_logic(start, end, _use_backup=True)
        except Exception as e:
            print(f"  Per-run archive failed ({e}) — trying MMSDM monthly archive...", flush=True)
            return _repeat_logic(start, end, _use_mmsdm=True)

    _month_compiler(
        datasource_fetch_function = _repeat_logic,
        start_date = start_date,
        end_date = end_date,
        datasource_file_path = datasource_file_path,
        fallback_fetch_function = _repeat_logic_backup,
        cache_dir = cache_dir,
    )


"""
Shared helpers for the bid datasources (8.1 / 8.2)

AEMO dropped the 5-minute "as-bid" _D tables (BIDPEROFFER_D / BIDDAYOFFER_D)
from the MMSDM historical archive around 2021-03; for those months only the
settlement versions (BIDPEROFFER / BIDDAYOFFER) remain. nemosis only knows the
_D table names, so it 404s and raises NoDataToReturn. These helpers download and
parse the settlement tables directly so the gap can be back-filled via the
_month_compiler fallback mechanism.
"""
_BID_BAND_COLS = [f"BANDAVAIL{i}" for i in range(1, 11)]
_BID_PRICE_COLS = [f"PRICEBAND{i}" for i in range(1, 11)]


def _download_mmsdm_dvd_zip(file_stem: str, start: str) -> list[bytes]:
    # Download a monthly AEMO MMSDM "DVD" settlement table, returning every part
    # as raw zip bytes. Large tables are split by AEMO into numbered parts
    # (e.g. BIDPEROFFER1/BIDPEROFFER2 from ~2022-07): the single-file table is
    # tried first, then numbered parts are collected until one 404s. Raises
    # HTTPError(404) when no part exists for the month.
    month_ts = pd.Timestamp(start).replace(day=1)
    year, month = month_ts.year, month_ts.month

    def _fetch(stem):
        filename = f"PUBLIC_DVD_{stem}_{year}{month:02d}010000.zip"
        url = (
            f"https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"
            f"{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA/{filename}"
        )
        print(f"    MMSDM archive: downloading settlement {stem} {year}-{month:02d}...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read()

    # Single-file table (older months).
    try:
        return [_fetch(file_stem)]
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    # Numbered split parts (BIDPEROFFER1, BIDPEROFFER2, ...).
    parts = []
    part_no = 1
    while True:
        try:
            parts.append(_fetch(f"{file_stem}{part_no}"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                break
            raise
        part_no += 1

    if not parts:
        raise urllib.error.HTTPError(
            None, 404,
            f"no settlement DVD found for {file_stem} {year}-{month:02d}",
            None, None,
        )
    return parts


def _parse_aemo_dvd_csv(zdata: bytes, table_id: str, want_cols, bidtype_filter="ENERGY") -> pd.DataFrame:
    # Parse an AEMO MMSDM DVD zip (C/I/D-record CSV) into a string DataFrame
    # holding only `want_cols`, optionally filtered to a single BIDTYPE. Data rows
    # are read with pandas' C parser in chunks, loading only the needed columns,
    # so a month of settlement bids (tens/hundreds of millions of rows across all
    # bid types and versions) is parsed quickly with bounded memory rather than
    # via a per-row Python loop.
    import csv as _csv

    table_id = table_id.upper()
    want_up = [c.upper() for c in want_cols]
    frames = []
    with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
        for name in zf.namelist():
            if not name.upper().endswith(".CSV"):
                continue

            # First pass: read only the "I" header record for this table to map
            # column names to positions (AEMO fields start at index 4; a "D"
            # data row's values line up with the "I" row's header names).
            header = None
            resolved_table = None
            fallback_header = None
            fallback_table = None
            with zf.open(name) as f:
                text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                for line in text:
                    if line[:1] != "I":
                        continue
                    parts = next(_csv.reader([line]))
                    if len(parts) < 3:
                        continue
                    this_table = parts[2].upper()
                    if fallback_header is None:
                        fallback_header = [c.upper() for c in parts[4:]]
                        fallback_table = this_table
                    if this_table == table_id:
                        header = [c.upper() for c in parts[4:]]
                        resolved_table = this_table
                        break
            if header is None:
                # AEMO renamed the settlement record in the split parts (e.g.
                # BIDPEROFFER1/2); fall back to the file's sole table when the
                # expected record name isn't present.
                header = fallback_header
                resolved_table = fallback_table
            if not header:
                continue

            width = 4 + len(header)
            pos = {c: 4 + i for i, c in enumerate(header)}
            bidtype_pos = pos.get("BIDTYPE")
            want_pos = {c: pos[c] for c in want_up if c in pos}
            usecols = sorted({0, 2, *want_pos.values()} |
                             ({bidtype_pos} if bidtype_pos is not None else set()))

            # Second pass: C-parser, only the columns we need, filtered per chunk.
            with zf.open(name) as f:
                reader = pd.read_csv(
                    f, header=None, names=list(range(width)), usecols=usecols,
                    dtype=str, engine="c", on_bad_lines="skip",
                    chunksize=1_000_000,
                )
                scanned = kept = 0
                for chunk in reader:
                    scanned += len(chunk)
                    chunk = chunk[(chunk[0] == "D") & (chunk[2].str.upper() == resolved_table)]
                    if bidtype_filter and bidtype_pos is not None:
                        chunk = chunk[chunk[bidtype_pos] == bidtype_filter]
                    if chunk.empty:
                        continue
                    kept += len(chunk)
                    frames.append(pd.DataFrame({
                        c: (chunk[want_pos[c]] if c in want_pos else None)
                        for c in want_up
                    }))
                    if scanned % 10_000_000 < 1_000_000:
                        print(f"      parsing {table_id}: {scanned:,} rows scanned, "
                              f"{kept:,} kept...", flush=True)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=want_up)


def _clean_bid_availability(API_response: pd.DataFrame) -> pd.DataFrame:
    API_response.columns = [c.upper() for c in API_response.columns]
    API_response = API_response.rename(columns={"INTERVAL_DATETIME": "SETTLEMENTDATE"})

    num_cols = ["MAXAVAIL"] + _BID_BAND_COLS

    # nemosis is called with parse_data_types=False and reads feather, so every
    # column comes back as (arrow-backed) strings; the settlement fallback also
    # yields string columns. Build a clean NumPy-backed frame with genuine float
    # columns so the groupby .mean() below works — assigning converted values back
    # into an arrow frame kept the str dtype.
    work = pd.DataFrame({
        "SETTLEMENTDATE": pd.to_datetime(API_response["SETTLEMENTDATE"]),
        "DUID": API_response["DUID"].astype(str),
    })
    for col in num_cols:
        work[col] = pd.to_numeric(API_response[col].astype(str), errors="coerce").astype("float64")
    work[num_cols] = work[num_cols].fillna(0.0)

    agg = work.groupby(["SETTLEMENTDATE", "DUID"])[num_cols].mean()

    # {DUID}_maxavail columns
    maxavail = agg["MAXAVAIL"].unstack("DUID")
    maxavail.columns = [f"{duid}_maxavail" for duid in maxavail.columns]
    maxavail = maxavail.resample("5min").mean().fillna(0)

    # {DUID}_bands — resample numerically first, then convert to comma-separated strings
    # This avoids slow string-based resampling
    band_data = agg[_BID_BAND_COLS].unstack("DUID").resample("5min").last()

    duids = sorted(band_data.columns.get_level_values("DUID").unique())
    bands_dict = {}
    for duid in duids:
        duid_cols = band_data.xs(duid, axis=1, level="DUID").fillna(0).astype(int).astype(str)
        bands_dict[f"{duid}_bands"] = duid_cols.iloc[:, 0].str.cat(duid_cols.iloc[:, 1:], sep=",")
    bands = pd.DataFrame(bands_dict, index=band_data.index)

    result = pd.concat([maxavail, bands], axis=1)

    all_duids = sorted(c.rsplit("_maxavail", 1)[0] for c in maxavail.columns)
    ordered = [col for duid in all_duids for col in (f"{duid}_maxavail", f"{duid}_bands") if col in result.columns]
    result = result[ordered]
    result.index.name = "Date"
    return result


def _clean_bid_prices(API_response: pd.DataFrame) -> pd.DataFrame:
    API_response.columns = [c.upper() for c in API_response.columns]

    # See _clean_bid_availability: build a NumPy-backed float frame so the
    # groupby .mean() works regardless of whether the source was nemosis feather
    # or the settlement-table fallback (both return string columns).
    work = pd.DataFrame({
        "SETTLEMENTDATE": pd.to_datetime(API_response["SETTLEMENTDATE"]),
        "DUID": API_response["DUID"].astype(str),
    })
    for col in _BID_PRICE_COLS:
        work[col] = pd.to_numeric(API_response[col].astype(str), errors="coerce").astype("float64")
    work[_BID_PRICE_COLS] = work[_BID_PRICE_COLS].fillna(0.0)

    agg = work.groupby(["SETTLEMENTDATE", "DUID"])[_BID_PRICE_COLS].mean()

    # Resample numerically first (ffill daily→5min), then convert to comma-separated strings
    price_data = agg.unstack("DUID").resample("5min").ffill()

    duids = sorted(price_data.columns.get_level_values("DUID").unique())
    prices_dict = {}
    for duid in duids:
        duid_cols = price_data.xs(duid, axis=1, level="DUID").fillna(0).astype(int).astype(str)
        prices_dict[f"{duid}_prices"] = duid_cols.iloc[:, 0].str.cat(duid_cols.iloc[:, 1:], sep=",")
    result = pd.DataFrame(prices_dict, index=price_data.index)

    result.index.name = "Date"
    return result


"""
Datasource 8.1
"""
def _bid_availability(start: str, end: str, cache_dir="Pre_processing/temporary_cache") -> pd.DataFrame:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _API_call():
        return nemosis.dynamic_data_compiler(
            start_time=start, end_time=end,
            table_name="BIDPEROFFER_D",
            raw_data_location=str(cache_dir),
            select_columns=["INTERVAL_DATETIME", "DUID", "BIDTYPE", "MAXAVAIL"] + _BID_BAND_COLS,
            filter_cols=["BIDTYPE"], filter_values=[["ENERGY"]],
            fformat="feather", keep_csv=False, parse_data_types=False,
        )

    return _clean_bid_availability(_API_call())


def _bid_availability_fallback(start: str, end: str, cache_dir="Pre_processing/temporary_cache") -> pd.DataFrame:
    # Back-fill from the settlement BIDPEROFFER table (record name BIDOFFERPERIOD)
    # when nemosis can't find the retired BIDPEROFFER_D table for this month.
    parts = _download_mmsdm_dvd_zip("BIDPEROFFER", start)
    want = ["TRADINGDATE", "PERIODID", "DUID", "BIDTYPE", "MAXAVAIL"] + _BID_BAND_COLS
    raw = pd.concat(
        [_parse_aemo_dvd_csv(z, "BIDOFFERPERIOD", want) for z in parts],
        ignore_index=True,
    )
    if raw.empty:
        raise RuntimeError("settlement BIDPEROFFER contained no ENERGY rows")

    # The settlement table has no INTERVAL_DATETIME; reconstruct the interval-
    # ending timestamp from TRADINGDATE + PERIODID. The NEM trading day starts at
    # 04:00, so period p ends at 04:00 + p*step, where step is 5 min post-5MS
    # (max period 288) or 30 min before it (max period 48).
    period = pd.to_numeric(raw["PERIODID"], errors="coerce")
    step = pd.Timedelta(minutes=5) if period.max() > 48 else pd.Timedelta(minutes=30)
    raw["INTERVAL_DATETIME"] = (
        pd.to_datetime(raw["TRADINGDATE"]) + pd.Timedelta(hours=4) + period * step
    )
    raw = raw.drop(columns=["TRADINGDATE", "PERIODID"])
    print(f"    MMSDM archive: parsed {len(raw):,} settlement BIDPEROFFER rows.", flush=True)
    return _clean_bid_availability(raw)


"""
Datasource 8.2
"""
def _bid_prices(start: str, end: str, cache_dir="Pre_processing/temporary_cache") -> pd.DataFrame:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _API_call():
        return nemosis.dynamic_data_compiler(
            start_time=start, end_time=end,
            table_name="BIDDAYOFFER_D",
            raw_data_location=str(cache_dir),
            select_columns=["SETTLEMENTDATE", "DUID", "BIDTYPE"] + _BID_PRICE_COLS,
            filter_cols=["BIDTYPE"], filter_values=[["ENERGY"]],
            fformat="feather", keep_csv=False, parse_data_types=False,
        )

    return _clean_bid_prices(_API_call())


def _bid_prices_fallback(start: str, end: str, cache_dir="Pre_processing/temporary_cache") -> pd.DataFrame:
    # Back-fill from the settlement BIDDAYOFFER table when nemosis can't find the
    # retired BIDDAYOFFER_D table for this month. The settlement table already
    # carries SETTLEMENTDATE + PRICEBAND1..10, so it maps straight through.
    parts = _download_mmsdm_dvd_zip("BIDDAYOFFER", start)
    want = ["SETTLEMENTDATE", "DUID", "BIDTYPE"] + _BID_PRICE_COLS
    raw = pd.concat(
        [_parse_aemo_dvd_csv(z, "BIDDAYOFFER", want) for z in parts],
        ignore_index=True,
    )
    if raw.empty:
        raise RuntimeError("settlement BIDDAYOFFER contained no ENERGY rows")
    print(f"    MMSDM archive: parsed {len(raw):,} settlement BIDDAYOFFER rows.", flush=True)
    return _clean_bid_prices(raw)
