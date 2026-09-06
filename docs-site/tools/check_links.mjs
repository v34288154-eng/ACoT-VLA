// Check every internal markdown link (like [x](/overview) or raw /generated/repo-tree)
// in the docs resolves to a real page or public asset.
// Usage (from docs-site/):  node tools/check_links.mjs
import { readdirSync, readFileSync, existsSync, statSync } from "node:fs";
import { join, dirname, relative } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const docsDir = join(here, "..", "docs");
const LINK_RE = /\]\((\/[^)\s]+)\)/g;

function collectMd(dir, acc = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith(".vitepress")) continue;
    const full = join(dir, entry.name);
    if (entry.isDirectory()) collectMd(full, acc);
    else if (entry.name.endsWith(".md")) acc.push(full);
  }
  return acc;
}

function resolveTarget(pageFile, target) {
  const clean = target.split("#")[0].split("?")[0];
  if (!clean.startsWith("/")) return { ok: true }; // relative links handled by vitepress
  const withoutSlash = clean.replace(/\/+$/, "");
  // public asset (figures etc.)
  const pub = join(docsDir, "public", ...withoutSlash.split("/"));
  if (existsSync(pub) && statSync(pub).isFile()) return { ok: true };
  // page (cleanUrls => /overview maps to docs/overview.md)
  const page = join(docsDir, ...withoutSlash.split("/")) + ".md";
  if (existsSync(page)) return { ok: true };
  return { ok: false, target, from: relative(docsDir, pageFile) };
}

const files = collectMd(docsDir);
const problems = [];
let checked = 0;
for (const f of files) {
  const text = readFileSync(f, "utf8");
  for (const m of text.matchAll(LINK_RE)) {
    checked++;
    const r = resolveTarget(f, m[1]);
    if (!r.ok) problems.push(`${r.from}: broken link -> ${r.target}`);
  }
}
console.log(`checked ${checked} internal links/paths across ${files.length} files: ${problems.length} problem(s)`);
for (const p of problems.slice(0, 40)) console.log(`  ${p}`);
process.exit(problems.length ? 1 : 0);
