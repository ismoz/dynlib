# src/dynlib/mkdocs_helpers.py
"""Helper utilities for generating documentation for mkdocs."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "src" / "dynlib" / "models"
LANGS = ("en", "tr")
PROJECT_FILES = [
    ("Changelog", ROOT / "CHANGELOG.md"),
    ("Issues", ROOT / "ISSUES.md"),
    ("TODO", ROOT / "TODO.md"),
]

_L10N = {
    "en": {
        "models_title": "Built-in model library",
        "map_models": "Map models",
        "ode_models": "ODE models",
        "map_index_title": "Map models",
        "ode_index_title": "ODE models",
        "source": "Source",
    },
    "tr": {
        "models_title": "Yerleşik model kütüphanesi",
        "map_models": "Harita modelleri",
        "ode_models": "ODE modelleri",
        "map_index_title": "Harita modelleri",
        "ode_index_title": "ODE modelleri",
        "source": "Kaynak",
    },
}


def _t(lang: str, key: str) -> str:
    # Fail fast if a language/key is missing (keeps things tidy).
    return _L10N[lang][key]


def slug(path: Path) -> str:
    return path.stem.replace("_", "-")


def _write(docs_root: Path, relpath: str, content: str) -> None:
    out = docs_root / relpath
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")


def _generate_models_section(docs_root: Path, lang: str) -> None:
    models_dir = f"{lang}/reference/models"

    _write(
        docs_root,
        f"{models_dir}/index.md",
        (
            f"# {_t(lang, 'models_title')}\n\n"
            f"- [{_t(lang, 'map_models')}](map/index.md)\n"
            f"- [{_t(lang, 'ode_models')}](ode/index.md)\n"
        ),
    )

    for kind in ("map", "ode"):
        kind_dir = f"{models_dir}/{kind}"
        names: list[str] = []

        for toml_file in sorted((MODELS / kind).glob("*.toml")):
            name = slug(toml_file)
            names.append(name)

            toml_text = toml_file.read_text(encoding="utf-8")

            doc_content = [
                f"# `{toml_file.name}`\n",
                f"{_t(lang, 'source')}: `src/dynlib/models/{kind}/{toml_file.name}`\n",
                "```toml\n",
                toml_text,
                "\n```\n",
            ]
            _write(docs_root, f"{kind_dir}/{name}.md", "".join(doc_content))

        index_title = _t(lang, "map_index_title" if kind == "map" else "ode_index_title")
        index_lines = [f"# {index_title}\n\n"]
        for name in names:
            index_lines.append(f"- [{name}]({name}.md)\n")
        _write(docs_root, f"{kind_dir}/index.md", "".join(index_lines))


def _generate_project_section(docs_root: Path, lang: str) -> None:
    project_dir = f"{lang}/project"

    for title, src_path in PROJECT_FILES:
        if not src_path.exists():
            continue

        slug_name = src_path.stem.lower()
        target_file = f"{project_dir}/{slug_name}.md"

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
        _write(docs_root, target_file, "".join(target_content))


def generate_model_docs(docs_root: Path) -> None:
    for lang in LANGS:
        _generate_models_section(docs_root, lang)
        _generate_project_section(docs_root, lang)
