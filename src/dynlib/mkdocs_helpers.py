"""Helper utilities for generating documentation artifacts for mkdocs."""
from __future__ import annotations

import shutil
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "src" / "dynlib" / "models"
LANGS = ("en", "tr")  # Generate docs for both locales.
PROJECT_FILES = [
    ("Changelog", ROOT / "CHANGELOG.md"),
    ("Issues", ROOT / "ISSUES.md"),
    ("TODO", ROOT / "TODO.md"),
]


def slug(path: Path) -> str:
    """Convert a model file name into a nav-friendly slug."""
    return path.stem.replace("_", "-")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _generate_models_section(target: Path) -> None:
    """Create the model index, kind pages, and literate nav inside a specific directory."""
    models_dir = target / "reference" / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    _write(models_dir / "index.md", "# Built-in model library\n\n- [Map models](map/index.md)\n- [ODE models](ode/index.md)\n")

    for kind in ("map", "ode"):
        kind_dir = models_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        names: list[str] = []

        for toml_file in sorted((MODELS / kind).glob("*.toml")):
            name = slug(toml_file)
            names.append(name)

            doc_path = kind_dir / f"{name}.md"
            toml_text = toml_file.read_text(encoding="utf-8")

            doc_content = [
                f"# `{toml_file.name}`\n",
                f"Source: `src/dynlib/models/{kind}/{toml_file.name}`\n",
                "```toml\n",
                toml_text,
                "\n```\n",
            ]

            _write(doc_path, "".join(doc_content))

        index_lines = [f"# {kind.upper()} models\n\n"]
        for name in names:
            index_lines.append(f"- [{name}]({name}.md)\n")
        _write(kind_dir / "index.md", "".join(index_lines))

    nav = mkdocs_gen_files.Nav()
    nav[("Overview",)] = "index.md"

    for kind in ("map", "ode"):
        nav[(kind.upper(), "Overview")] = f"{kind}/index.md"
        for toml_file in sorted((MODELS / kind).glob("*.toml")):
            name = slug(toml_file)
            nav[(kind.upper(), name)] = f"{kind}/{name}.md"

    _write(models_dir / "SUMMARY.md", "".join(nav.build_literate_nav()))


def _generate_project_section(target: Path) -> None:
    """Copy the shared project docs into the target directory."""
    project_dir = target / "project"
    for title, src_path in PROJECT_FILES:
        if not src_path.exists():
            continue
        slug_name = src_path.stem.lower()
        target_file = project_dir / f"{slug_name}.md"

        text = src_path.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            text += "\n"

        target_content = [
            f"# {title}\n\n",
            f"Source: `{src_path.name}`\n\n",
            "```text\n",
            text,
            "```\n",
        ]

        _write(target_file, "".join(target_content))


def generate_model_docs(docs_root: Path) -> None:
    """
    Generate the model and project reference pages inside the docs tree.

    Args:
        docs_root: Base directory where language subfolders live (typically ``docs``).
    """
    docs_root = Path(docs_root)
    targets = [docs_root]
    targets.extend(docs_root / lang for lang in LANGS)

    for target in targets:
        _generate_models_section(target)
        _generate_project_section(target)
