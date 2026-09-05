#!/usr/bin/env python3
"""Verify every `path/to/file.py:NNN` anchor used in the docs.

Rules enforced:
- Anchors must be repo-root-relative paths with a '/' and a line number
  (ranges like `config.py:1236-1240` and en-dash `–` ranges are accepted).
- The file must exist under the repo and the line number must be in range.
- A bare filename like `data_loader.py:599` is only accepted when it resolves
  to exactly one file in the scanned roots (a warning is printed otherwise).

Usage:
    python tools/check_anchors.py --repo <repo-root> [--docs <docs-site/docs>]
Exit code 0 = all anchors valid; 1 = violations found.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ANCHOR_RE = re.compile(
    r"(?<![\w./\-\\])((?:\w[\w./\-\\]*/)?[\w\-.]+\.py):(\d+)(?:[\-\u2013](\d+))?"
)
ROOTS = ("src/openpi", "packages/openpi-client/src/openpi_client",
         "scripts", "examples", "")


def collect_md_files(docs: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in docs.rglob("*.md") if ".vitepress" not in p.parts]


def resolve(rel: str, root: pathlib.Path) -> tuple[pathlib.Path | None, list[str]]:
    rel = rel.replace("\\", "/")
    candidates: list[pathlib.Path] = []
    if rel.startswith("/"):
        rel = rel.lstrip("/")
        candidates.append(root / rel)
    else:
        for r in ROOTS:
            if r:
                candidates.append(root / r / rel)
            else:
                candidates.append(root / rel)
    hits = sorted({c for c in candidates if c.exists()})
    if hits:
        return hits[0], []
    if "/" not in rel:
        # bare filename fallback: search all scanned roots + repo root
        name_hits: list[pathlib.Path] = []
        for r in ROOTS:
            base = root / r if r else root
            if base.is_dir():
                for c in base.glob(f"**/{rel}"):
                    if "third_party" not in c.parts and ".git" not in c.parts:
                        name_hits.append(c)
        unique = sorted({p.resolve() for p in name_hits})
        if len(unique) == 1:
            return unique[0], []
        if len(unique) > 1:
            return None, [f"ambiguous bare filename `{rel}` matches {len(unique)} files"]
    return None, []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[2])
    ap.add_argument("--docs", type=pathlib.Path, default=None)
    args = ap.parse_args()
    root = args.repo.resolve()
    docs = (args.docs or (root / "docs-site" / "docs")).resolve()

    problems: list[str] = []
    warnings: list[str] = []
    total = 0
    for md in collect_md_files(docs):
        text = md.read_text(encoding="utf-8", errors="replace")
        for m in ANCHOR_RE.finditer(text):
            rel, start_s, end_s = m.group(1), m.group(2), m.group(3)
            total += 1
            path_hit, notes = resolve(rel, root)
            for note in notes:
                problems.append(f"{md.relative_to(docs)}: {note}")
            if path_hit is None:
                problems.append(f"{md.relative_to(docs)}: anchor `{rel}:{start_s}` -> file not found")
                continue
            nlines = len(path_hit.read_text(encoding="utf-8", errors="replace").splitlines())
            start = int(start_s)
            end = int(end_s) if end_s else start
            if start < 1 or end < start or end > nlines:
                problems.append(
                    f"{md.relative_to(docs)}: anchor `{rel}:{start_s}` "
                    f"(range {end_s or start_s}) out of file bounds (file has {nlines} lines)"
                )
    for md in collect_md_files(docs):
        text = md.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"`([\w./\-]+\.py)`", text):
            rel = m.group(1)
            if rel in text:  # already counted above when followed by :NNN
                continue
            if resolve(rel, root)[0] is not None:
                continue
            warnings.append(f"{md.relative_to(docs)}: code path `{rel}` not resolvable in repo")

    print(f"checked {total} anchors across {len(collect_md_files(docs))} files")
    for w in warnings[:20]:
        print(f"  warn: {w}")
    if problems:
        print(f"{len(problems)} violation(s):")
        for p in problems[:40]:
            print(f"  error: {p}")
        return 1
    print("all anchors OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
