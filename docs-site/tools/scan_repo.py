#!/usr/bin/env python3
"""Generate a repo tree snapshot (Markdown) for the ACoT-VLA docs site.

Usage:
    python tools/scan_repo.py --repo <repo-root> --commit <git-sha>
Outputs: <repo>/docs-site/docs/generated/repo-tree.md
Pure stdlib; runnable without the project's uv environment.
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import sys

SKIP_DIRS = {
    ".git", "third_party", "node_modules", ".venv", "venv", "__pycache__",
    ".vitepress", "dist", ".idea", ".ruff_cache", ".pytest_cache",
    ".next", "checkpoints", "assets", ".pixi",
}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="output markdown path (default: <repo>/docs-site/docs/generated/repo-tree.md)")
    ap.add_argument("--commit", default="unknown", help="git commit the snapshot is based on")
    args = ap.parse_args()

    root = args.repo.resolve()
    if not (root / ".git").exists():
        print(f"error: {root} does not look like a git repo", file=sys.stderr)
        return 1
    out = args.out or (root / "docs-site" / "docs" / "generated" / "repo-tree.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 仓库目录快照（自动生成）",
        "",
        f"> 生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M}；基线 commit：`{args.commit}`。",
        "> 由 `python tools/scan_repo.py --repo <repo> --commit <sha>` 生成，请勿手改。",
        "",
        "```text",
        ".",
    ]

    def walk(d: pathlib.Path, prefix: str) -> None:
        try:
            entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return
        visible = [p for p in entries
                   if p.name not in SKIP_DIRS and p.suffix not in SKIP_SUFFIXES]
        for i, p in enumerate(visible):
            last = i == len(visible) - 1
            stem = "└── " if last else "├── "
            if p.is_dir():
                lines.append(f"{prefix}{stem}{p.name}/")
                walk(p, prefix + ("    " if last else "│   "))
            else:
                try:
                    sz = p.stat().st_size
                except OSError:
                    sz = 0
                size = f"{sz / 1024:.1f} KB" if sz < 1024 * 1024 else f"{sz / 1024 / 1024:.2f} MB"
                lines.append(f"{prefix}{stem}{p.name}   ({size})")

    walk(root, "")
    lines.append("```")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
