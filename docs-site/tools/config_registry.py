#!/usr/bin/env python3
"""Export the named-config registry from src/openpi/training/config.py as Markdown.

Finds the module-level `_CONFIGS = [TrainConfig(...), ...]` list via `ast` and
emits a table of: config name, source line, model class, data-factory class and
a short summary of literal model kwargs (e.g. action_horizon / pi05 overrides).

Usage:
    python tools/config_registry.py --repo <repo-root> --commit <git-sha>
Outputs: docs-site/docs/generated/config-registry.md
"""
from __future__ import annotations

import argparse
import ast
import datetime
import pathlib
import sys


def expr_name(node: ast.AST | None) -> str:
    """Render a dotted name chain for Call/Attribute/Name nodes."""
    if node is None:
        return "-"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = expr_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return expr_name(node.func)
    if isinstance(node, ast.Lambda):
        return "<lambda>"
    return type(node).__name__


def literal_repr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, str):
            return v if len(v) <= 60 else v[:57] + "..."
        return repr(v)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant):
        return repr(-node.operand.value if isinstance(node.op, ast.USub) else node.operand.value)
    return None


def model_summary(call: ast.Call) -> str:
    parts: list[str] = []
    for kw in call.keywords:
        if kw.arg is None:
            continue
        val = literal_repr(kw.value)
        if val is None:
            continue
        if kw.arg in ("dtype",):
            continue
        parts.append(f"{kw.arg}={val}")
    s = ", ".join(parts)
    return s if len(s) <= 110 else s[:107] + "..."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()

    root = args.repo.resolve()
    cfg = root / "src" / "openpi" / "training" / "config.py"
    if not cfg.exists():
        print(f"error: {cfg} not found", file=sys.stderr)
        return 1
    out = args.out or (root / "docs-site" / "docs" / "generated" / "config-registry.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    tree = ast.parse(cfg.read_text(encoding="utf-8"))
    configs: list[tuple[int, str, str, str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_CONFIGS" and isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if not isinstance(elt, ast.Call):
                            continue
                        kwargs = {kw.arg: kw.value for kw in elt.keywords if kw.arg is not None}
                        name = literal_repr(kwargs.get("name")) if isinstance(kwargs.get("name"), ast.Constant) else None
                        model_cls = expr_name(kwargs.get("model"))
                        data_cls = expr_name(kwargs.get("data"))
                        model_expr = kwargs.get("model")
                        summary = model_summary(model_expr) if isinstance(model_expr, ast.Call) else ""
                        configs.append((elt.lineno, str(name) if name is not None else "?", model_cls, data_cls, summary))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "_CONFIGS":
            # Typed container form (not used in this repo, kept for robustness).
            pass

    configs.sort(key=lambda r: r[0])
    counts: dict[str, int] = {}
    rows = []
    for line, name, model, data, summary in configs:
        rows.append(f"| `{name}` | {line} | {model} | {data} | {summary} |")
        for marker in ("ACOT", "Pi0", "Pi0FAST"):
            if marker in model:
                counts[marker] = counts.get(marker, 0) + 1
                break
        else:
            counts["other"] = counts.get("other", 0) + 1

    stat = " · ".join(f"{k}×{v}" for k, v in sorted(counts.items())) or "0"
    lines = [
        "# 命名配置注册表（自动生成）",
        "",
        f"> 来源：`src/openpi/training/config.py` 的 `_CONFIGS` 列表；生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M}；基线 commit：`{args.commit}`。",
        f"> 统计：共 {len(configs)} 个命名配置（按模型类粗略分类：{stat}）。运行时以 `get_config(name)` 查询，CLI 经 tyro 覆盖字段。",
        "> 由 `python tools/config_registry.py --repo <repo> --commit <sha>` 生成，请勿手改。",
        "",
        "| 配置名 | 行号 | 模型 | 数据工厂 | 模型关键覆盖 |",
        "|---|---|---|---|---|",
    ] + rows + [""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}: {len(configs)} configs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
