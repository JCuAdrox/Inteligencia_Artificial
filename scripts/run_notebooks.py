"""Ejecuta los notebooks del workshop con nbclient y guarda sus salidas.

Requisitos: NVIDIA_API_KEY en el entorno (nunca en archivos) y las dependencias
de requirements.txt. Las capturas de Gradio se toman con Playwright cuando
TOMAR_CAPTURAS=1 (este script lo activa).

Uso:
    python scripts/run_notebooks.py                 # los tres notebooks
    python scripts/run_notebooks.py 04_challenges   # solo los que coincidan
"""
import os
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "02b_rag_app_nvidia_nim_colab.ipynb",
    "03_rag_workshop_papers_nvidia_nim_colab.ipynb",
    "04_challenges.ipynb",
]
KEY_PATTERN = re.compile(r"nvapi-[A-Za-z0-9_\-]{8,}")


def run(name: str) -> bool:
    path = ROOT / "notebooks" / name
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(path.parent)}},
    )
    start = time.time()
    client.execute()
    nbformat.write(nb, path)

    errors = [
        out for cell in nb.cells if cell.cell_type == "code"
        for out in cell.get("outputs", []) if out.get("output_type") == "error"
    ]
    leaked = KEY_PATTERN.search(path.read_text(encoding="utf-8"))
    print(f"{name}: {time.time() - start:.0f}s, {len(errors)} celdas con error"
          + (" — ¡ALERTA: aparece una API key!" if leaked else ""))
    for err in errors:
        print(f"   {err['ename']}: {err['evalue'][:200]}")
    return not leaked


if __name__ == "__main__":
    if not os.getenv("NVIDIA_API_KEY"):
        sys.exit("Falta NVIDIA_API_KEY en el entorno.")
    os.environ.setdefault("TOMAR_CAPTURAS", "1")
    selected = [n for n in NOTEBOOKS if not sys.argv[1:] or any(a in n for a in sys.argv[1:])]
    ok = all(run(name) for name in selected)
    sys.exit(0 if ok else 1)
