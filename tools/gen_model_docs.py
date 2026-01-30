# tools/gen_model_docs.py
""" 
Wrapper for mkdocs doc generator (for models and changelog).

This script should be run manually when you want to create docs 
or when you made changes to the built-in models or changelog.

For Github Actions prepare a workflow that runs this script before 
building the docs (see .github/workflows/docs.yml).
"""
from pathlib import Path
from dynlib.mkdocs_helpers import generate_model_docs

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    generate_model_docs(ROOT / "docs")
