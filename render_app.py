import os
import json
from pathlib import Path

import dash

NOTEBOOK_FILE = Path(__file__).with_name("final_Stock_git_ready copy.ipynb")


def _normalize_source(source):
    if isinstance(source, list):
        return "".join(source)
    if isinstance(source, str):
        return source
    return ""


def _clean_cell_code(code):
    cleaned_lines = []
    for line in code.splitlines():
        stripped = line.strip()

        # Skip notebook magics and shell commands not valid in script execution.
        if stripped.startswith("%") or stripped.startswith("!"):
            continue

        # Skip local run commands so Render/Gunicorn owns the process.
        if "app.run_server(" in stripped or "app.run(" in stripped:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _execute_notebook_code(notebook_path):
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    with notebook_path.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    exec_namespace = {"__name__": "__main__"}

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = _normalize_source(cell.get("source", ""))
        code = _clean_cell_code(source)

        if not code.strip():
            continue

        try:
            exec(code, exec_namespace)
        except Exception as exc:
            # Keep going to preserve notebook-style execution behavior.
            print(f"Warning: skipped a failing cell during startup: {exc}")

    return exec_namespace


namespace = _execute_notebook_code(NOTEBOOK_FILE)

if "app" not in namespace or not isinstance(namespace["app"], dash.Dash):
    raise RuntimeError("Dash app object was not created from notebook code.")

app = namespace["app"]
server = app.server


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8060"))
    app.run(host="0.0.0.0", port=port, debug=False)
