"""Execute the notebook in the current Python environment, logging progress."""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


class ProgressClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return super().process_message(msg, cell, cell_index)


if __name__ == "__main__":
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    path = Path(__file__).resolve().parent / "anomaly_detection_project_question.ipynb"
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    def started(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Running cell {cell_index}", flush=True)
    client = ProgressClient(notebook, km=manager, timeout=None,
                            resources={"metadata": {"path": str(path.parent)}},
                            on_cell_start=started)
    try:
        client.execute()
        print("SUCCESS: all notebook cells executed.", flush=True)
    finally:
        nbformat.write(notebook, path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        print("Saved notebook outputs.", flush=True)
