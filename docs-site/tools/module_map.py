#!/usr/bin/env python3
"""Generate a module map (classes/functions with line numbers) from the repo.

Scans <repo>/src/openpi, <repo>/packages, <repo>/scripts and <repo>/examples,
parses every public .py module with `ast` and writes a Markdown map:
    docs-site/docs/generated/module-map.md

Usage:
    python tools/module_map.py --repo <repo-root> --commit <git-sha>
Pure stdlib; no project dependencies needed.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import pathlib
import sys

SCAN_DIRS = ("src/openpi", "packages", "scripts", "examples")
SKIP_PARTS = {".git", "third_party", "node_modules", "__pycache__", ".venv"}


def first_doc_line(node: ast.AST) -> str:
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    first = doc.strip().splitlines()[0].strip()
    return first[:120]


def collect(file: pathlib.Path, out: list[str]) -> tuple[int, int]:
    try:
        tree = ast.parse(file.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as e:
        out.append(f"<!-- parse error in {file}: {e} -->")
        return 0, 0
    classes = funcs = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            funcs += 1
            doc = first_doc_line(node)
            out.append(f"- `{node.name}` (行 {node.lineno}){(' — ' + doc) if doc else ''}")
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            classes += 1
            doc = first_doc_line(node)
            out.append(f"- `{node.name}` (行 {node.lineno}){(' — ' + doc) if doc else ''}")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                    out.append(f"    - `{child.name}` (行 {child.lineno})")
    return classes, funcs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()

    root = args.repo.resolve()
    out = args.out or (root / "docs-site" / "docs" / "generated" / "module-map.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    total_files = total_classes = total_funcs = 0
    body: list[str] = []
    for rel in SCAN_DIRS:
        base = root / rel
        if not base.exists():
            continue
        for f in sorted(base.rglob("*.py")):
            if any(part in SKIP_PARTS for part in f.parts):
                continue
            name = f.name
            if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py" or name == "__init__.py":
                continue
            relpath = f.relative_to(root).as_posix()
            body.append(f"\n### {relpath}")
            before = len(body)
            c, fn = collect(f, body)
            if len(body) == before:  # no public symbols
                body.pop()
                continue
            total_files += 1
            total_classes += c
            total_funcs += fn

    lines = [
        "# 模块地图（自动生成：类/函数 → 文件:行号）",
        "",
        f"> 生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M}；基线 commit：`{args.commit}`。",
        f"> 覆盖 {SCAN_DIRS} 下的公开符号：{total_files} 个模块、{total_classes} 个类、{total_funcs} 个函数/方法。",
        "> 由 `python tools/module_map.py --repo <repo> --commit <sha>` 生成，请勿手改。",
    ] + body + [""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}: {total_files} modules, {total_classes} classes, {total_funcs} funcs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
