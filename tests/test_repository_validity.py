"""Repository-wide syntax and import checks; no model calls or ingestion runs."""
import ast
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIRS = ("src", "ingestion", "evaluation")


def test_repository_python_syntax():
    for directory in (*IMPLEMENTATION_DIRS, "tests"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def test_implementation_modules_import_without_execution(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for directory in IMPLEMENTATION_DIRS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            relative = path.relative_to(ROOT).with_suffix("")
            parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
            importlib.import_module(".".join(parts))


def test_persistent_path_text_io_declares_encoding():
    for directory in IMPLEMENTATION_DIRS:
        for path in (ROOT / directory).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in {"read_text", "write_text"}):
                    assert any(k.arg == "encoding" for k in node.keywords), (
                        f"{path.relative_to(ROOT)}:{node.lineno} relies on platform encoding")
