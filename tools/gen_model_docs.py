"""
Convenience wrapper to generate model docs from the command line.

Run this script to regenerate the model documentation in the `docs/` directory.
'mkdocs build' and 'mkdocs serve' does not seem to automatically regenerate.
"""

from pathlib import Path

from dynlib.mkdocs_helpers import generate_model_docs

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    generate_model_docs(ROOT / "docs")
