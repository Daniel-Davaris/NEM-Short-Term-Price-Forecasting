import concurrent.futures
import math
import os
import pickle
import psutil
import sys
from tqdm import tqdm


_WORKER_FUNCTION = None
_WORKER_STDOUT = None
_WORKER_STDERR = None


# Initialises each worker process: stores the target function in a global and optionally silences stdout/stderr.
def _init_process_worker(function, silence_worker_output):
    global _WORKER_FUNCTION, _WORKER_STDOUT, _WORKER_STDERR
    _WORKER_FUNCTION = function

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    if silence_worker_output:
        _WORKER_STDOUT = open(os.devnull, "w")
        _WORKER_STDERR = open(os.devnull, "w")
        sys.stdout = _WORKER_STDOUT
        sys.stderr = _WORKER_STDERR


# Entry point called by each worker process — delegates to the globally stored function.
def _run_process_worker_item(item):
    return _WORKER_FUNCTION(item)


# Runs function(item) N times sequentially, returns results and average RSS growth per call.
def _probe(function, items):
    proc = psutil.Process(os.getpid())
    rss_before = int(proc.memory_info().rss)
    results = [function(item) for item in items]
    delta_per_item = max(proc.memory_info().rss - rss_before, 0) // len(items)
    return results, delta_per_item


# Runs function over every item in data. On Windows always sequential (1 worker).
# On Linux probes memory growth and sizes the worker pool accordingly.
def run_parallel(function, data):
    data = list(data)
    if not data:
        return []

    is_windows = sys.platform == "win32"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    if is_windows:
        print(f"os=windows cpu_count={os.cpu_count()} max_assigned_workers=1")
        results = []
        for item in tqdm(data, desc="Processing..", unit="item", dynamic_ncols=True, position=0, leave=True):
            results.append(function(item))
        return results

    # Linux: probe then parallelise with ProcessPoolExecutor
    picklable = True
    try:
        pickle.dumps(function, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        picklable = False

    n_probe = min(3, len(data))
    probe_results, peak_delta = _probe(function, data[:n_probe])

    available_memory = int(psutil.virtual_memory().available)
    cpu_workers = max(1, int(math.floor(os.cpu_count() * 0.95)))
    MIN_RELIABLE_DELTA = 100 * 1024 * 1024
    if peak_delta >= MIN_RELIABLE_DELTA:
        ram_workers = max(1, int(available_memory * 0.95) // peak_delta)
        max_workers = max(1, min(cpu_workers, ram_workers, len(data) - n_probe))
    else:
        max_workers = 1

    print(
        f"os=linux cpu_count={os.cpu_count()} available_memory_gb={available_memory / 1024**3:.2f} "
        f"peak_delta_gb={peak_delta / 1024**3:.2f} max_assigned_workers={max_workers}"
    )

    executor_class = concurrent.futures.ProcessPoolExecutor if picklable else concurrent.futures.ThreadPoolExecutor
    executor_kwargs = {"max_workers": max_workers}
    if picklable:
        executor_kwargs["initializer"] = _init_process_worker
        executor_kwargs["initargs"] = (function, True)

    with executor_class(**executor_kwargs) as executor:
        if picklable:
            futures = {executor.submit(_run_process_worker_item, item): i + n_probe for i, item in enumerate(data[n_probe:])}
        else:
            futures = {executor.submit(function, item): i + n_probe for i, item in enumerate(data[n_probe:])}

        results = [None] * len(data)
        for i, r in enumerate(probe_results):
            results[i] = r
        progress = tqdm(total=len(data), desc="Processing..", unit="item", dynamic_ncols=True, position=0, leave=True)
        progress.update(n_probe)
        try:
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
                progress.update(1)
        finally:
            progress.close()

        return results