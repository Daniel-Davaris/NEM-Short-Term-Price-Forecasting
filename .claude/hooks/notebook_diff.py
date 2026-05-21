#!/usr/bin/env python3
"""
PreToolUse hook for NotebookEdit.

Signals the VS Code extension about the incoming change, then exits 0
so Claude Code applies the edit normally (no "Failed" in the chat).
The extension intercepts the notebook change event and overlays the
red/green diff view with Keep / Undo buttons.
"""
import json
import os
import sys
import urllib.request
import urllib.error

PORT = 43127


def find_cell(cells, cell_id):
    for i, cell in enumerate(cells):
        cid = cell.get("id", "")
        if cid == cell_id or cid.startswith(cell_id) or cell_id.startswith(cid):
            return i, cell
    return None, None


def source_to_str(source):
    if isinstance(source, list):
        return "".join(source)
    return str(source) if source else ""


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if data.get("tool_name") != "NotebookEdit":
        sys.exit(0)

    inp = data.get("tool_input", {})
    notebook_path = inp.get("notebook_path") or inp.get("path", "")
    cell_id       = inp.get("cell_id", "")
    new_source_raw = inp.get("new_source")
    old_str        = inp.get("old_str")
    new_str        = inp.get("new_str")

    if not notebook_path or not os.path.isfile(notebook_path):
        sys.exit(0)

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception:
        sys.exit(0)

    cell_idx, cell = find_cell(nb.get("cells", []), cell_id)
    if cell is None:
        sys.exit(0)

    old_source_str = source_to_str(cell.get("source", ""))

    if new_source_raw is not None:
        new_source_str = source_to_str(new_source_raw)
    elif old_str is not None and new_str is not None:
        new_source_str = old_source_str.replace(old_str, new_str, 1)
    else:
        sys.exit(0)

    # Notify the extension about the incoming change.
    # We wait for the response (extension just stores the data and replies
    # immediately), then exit 0 so Claude Code applies the edit normally.
    payload = json.dumps({
        "notebook_path": notebook_path,
        "cell_id":       cell_id,
        "old_source":    old_source_str,
        "new_source":    new_source_str,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/incoming-change",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except urllib.error.URLError:
        # Extension not running — just let Claude Code apply normally.
        pass
    except Exception:
        pass

    # Exit 0 → Claude Code proceeds with the edit (no "Failed" in chat).
    sys.exit(0)


main()
