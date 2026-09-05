// Validate every ```mermaid fence in the docs with mermaid.parse().
// Usage (from docs-site/):  node tools/check_mermaid.mjs
// Exit code 0 = all diagrams parse; 1 = at least one syntax error.
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";

// mermaid's sanitizer needs a DOM, so provide a minimal one before importing it.
const dom = new JSDOM("<!DOCTYPE html><html><body></body></html>", { url: "http://localhost/" });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.self = dom.window;
globalThis.DOMPurify = undefined; // let mermaid create its own from the window above

const here = fileURLToPath(new URL(".", import.meta.url));
const docsDir = join(here, "..", "docs");
const FENCE_RE = /```mermaid\n([\s\S]*?)```/g;

function collectMd(dir, acc = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith(".vitepress")) continue;
    const full = join(dir, entry.name);
    if (entry.isDirectory()) collectMd(full, acc);
    else if (entry.name.endsWith(".md")) acc.push(full);
  }
  return acc;
}

const files = collectMd(docsDir);
const diagrams = [];
for (const f of files) {
  const text = readFileSync(f, "utf8");
  for (const m of text.matchAll(FENCE_RE)) {
    diagrams.push({ file: relative(docsDir, f), code: m[1] });
  }
}

const mermaid = (await import("mermaid")).default;
mermaid.initialize({ startOnLoad: false, securityLevel: "loose" });

let failures = 0;
for (const d of diagrams) {
  try {
    await mermaid.parse(d.code);
  } catch (e) {
    failures += 1;
    console.error(`MERMAID ERROR in ${d.file}: ${e.message?.split("\n")[0] ?? e}`);
  }
}
console.log(`checked ${diagrams.length} mermaid diagrams in ${files.length} files: ${failures} failure(s)`);
process.exit(failures ? 1 : 0);
