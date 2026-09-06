<script setup lang="ts">
/**
 * ImageViewer —— 全站图片/图表浮层查看器
 * - 点击正文 <img> 或 Mermaid 渲染出的 svg → 浮空大图显示
 * - 工具栏：放大 / 缩小 / 百分比(点击回 100%) / 复制图片 / 关闭
 * - 缩放范围 10% ~ 700%；滚轮缩放；拖拽平移；Esc 或点击空白关闭
 * - 查看器打开时支持 Ctrl+V 粘贴剪贴板图片直接查看
 */
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

const MIN_SCALE = 0.1; // 10%
const MAX_SCALE = 7.0; // 700%

const open = ref(false);
const mode = ref<"img" | "svg">("img");
const url = ref("");
const naturalW = ref(1);
const naturalH = ref(1);
const scale = ref(1);
const tx = ref(0);
const ty = ref(0);
const toast = ref("");
const dragging = ref(false);

const wrapEl = ref<HTMLDivElement | null>(null);
const stageEl = ref<HTMLDivElement | null>(null);

let svgEl: SVGSVGElement | null = null;
let drag: { sx: number; sy: number; tx: number; ty: number } | null = null;
let moved = false;
let pastedUrl = "";

const STAGE_PAD = 24;

/* ---------- 尺寸与变换 ---------- */

function stageSize() {
  return { w: window.innerWidth, h: window.innerHeight };
}

function contentBox() {
  return {
    cw: naturalW.value * scale.value,
    ch: naturalH.value * scale.value,
    ...stageSize(),
  };
}

function applySvgSize() {
  if (svgEl && mode.value === "svg") {
    svgEl.style.width = `${naturalW.value * scale.value}px`;
    svgEl.style.height = "auto";
    svgEl.style.maxWidth = "none";
  }
}

function clampPan() {
  const { cw, ch, w, h } = contentBox();
  const mx = cw > w ? (cw - w) / 2 + STAGE_PAD : 0;
  const my = ch > h ? (ch - h) / 2 + STAGE_PAD : 0;
  if (mx === 0) tx.value = 0;
  else tx.value = Math.min(mx, Math.max(-mx, tx.value));
  if (my === 0) ty.value = 0;
  else ty.value = Math.min(my, Math.max(-my, ty.value));
}

watch(scale, () => {
  applySvgSize();
  clampPan();
});

function zoom(factor: number) {
  scale.value = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale.value * factor));
}

function zoomIn() {
  zoom(1.25);
}
function zoomOut() {
  zoom(1 / 1.25);
}
function resetScale() {
  scale.value = 1;
  clampPan();
}

/* ---------- 打开 / 关闭 ---------- */

function prepareOpen() {
  document.body.style.overflow = "hidden";
  open.value = true;
  // 初始缩放：尽量铺满但不超过 100%
  const { w, h } = stageSize();
  const fit = Math.min((w * 0.9) / naturalW.value, (h * 0.8) / naturalH.value, 1);
  scale.value = Math.max(MIN_SCALE, fit);
  tx.value = 0;
  ty.value = 0;
  dragging.value = false;
}

function openImg(src: string, w: number, h: number) {
  cleanupSvg();
  mode.value = "img";
  url.value = src;
  naturalW.value = Math.max(1, w);
  naturalH.value = Math.max(1, h);
  prepareOpen();
}

async function openSvg(node: SVGSVGElement) {
  cleanupSvg();
  mode.value = "svg";
  url.value = "";
  const vb = node.viewBox?.baseVal;
  const rect = node.getBoundingClientRect();
  naturalW.value = Math.max(1, Math.round(vb?.width || rect.width || 800));
  naturalH.value = Math.max(1, Math.round(vb?.height || rect.height || 600));
  prepareOpen();
  // 浮层挂载后再插入矢量克隆（放大不失真）
  await nextTick();
  const clone = node.cloneNode(true) as SVGSVGElement;
  clone.removeAttribute("class");
  clone.removeAttribute("style");
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  wrapEl.value?.appendChild(clone);
  svgEl = clone;
  applySvgSize();
}

function cleanupSvg() {
  if (svgEl) {
    svgEl.remove();
    svgEl = null;
  }
}

function close() {
  open.value = false;
  cleanupSvg();
  document.body.style.overflow = "";
  if (pastedUrl) {
    URL.revokeObjectURL(pastedUrl);
    pastedUrl = "";
  }
  drag = null;
}

/* ---------- 复制 ---------- */

async function writeClip(entries: { type: string; blob: Blob }[]) {
  const items = entries.map((e) => new (window as any).ClipboardItem({ [e.type]: e.blob }));
  await navigator.clipboard.write(items);
}

function notify(msg: string) {
  toast.value = msg;
  window.setTimeout(() => (toast.value = ""), 2400);
}

async function copyImage() {
  try {
    if (mode.value === "img") {
      const resp = await fetch(url.value);
      if (!resp.ok) throw new Error(`图片加载失败 ${resp.status}`);
      const blob = await resp.blob();
      await writeClip([{ type: blob.type || "image/png", blob }]);
      notify("✅ 已复制图片，可直接粘贴到文档/聊天");
    } else {
      if (!svgEl) throw new Error("SVG 不存在");
      try {
        const s = svgEl.cloneNode(true) as SVGSVGElement;
        s.setAttribute("xmlns", "http://www.w3.org/2000/svg");
        if (!s.getAttribute("xmlns:xlink")) {
          s.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
        }
        const xml = new XMLSerializer().serializeToString(s);
        const svgUrl = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(xml);
        const img = new Image();
        await new Promise<void>((res, rej) => {
          img.onload = () => res();
          img.onerror = () => rej(new Error("SVG 无法栅格化"));
          img.src = svgUrl;
        });
        const canvas = document.createElement("canvas");
        const k = 2; // 2x 栅格化更清晰
        canvas.width = Math.round(naturalW.value * k);
        canvas.height = Math.round(naturalH.value * k);
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Canvas 不可用");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        const blob: Blob = await new Promise((res, rej) =>
          canvas.toBlob((b) => (b ? res(b) : rej(new Error("PNG 编码失败"))), "image/png")
        );
        await writeClip([{ type: "image/png", blob }]);
        notify("✅ 已复制为 PNG 图片");
      } catch {
        // 浏览器限制（如 html 标签无法栅格化）→ 回退为复制 SVG 矢量代码
        const xml = new XMLSerializer().serializeToString(svgEl);
        const html = `<svg xmlns="http://www.w3.org/2000/svg">${xml}</svg>`;
        await writeClip([
          { type: "text/html", blob: new Blob([html], { type: "text/html" }) },
          { type: "text/plain", blob: new Blob([xml], { type: "text/plain" }) },
        ]);
        notify("⚠️ PNG 转换受限，已复制 SVG 矢量代码（可粘贴到支持 SVG 的编辑器）");
      }
    }
  } catch (e: any) {
    notify(`❌ 复制失败：${e?.message || e}`);
  }
}

/* ---------- 交互事件 ---------- */

function onStagePointerDown(e: PointerEvent) {
  if (e.button !== 0) return;
  if ((e.target as HTMLElement).closest(".vz-toolbar")) return;
  dragging.value = true;
  moved = false;
  drag = { sx: e.clientX, sy: e.clientY, tx: tx.value, ty: ty.value };
  (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
}

function onStagePointerMove(e: PointerEvent) {
  if (!drag) return;
  const dx = e.clientX - drag.sx;
  const dy = e.clientY - drag.sy;
  if (Math.abs(dx) > 3 || Math.abs(dy) > 3) moved = true;
  tx.value = drag.tx + dx;
  ty.value = drag.ty + dy;
  clampPan();
}

function onStagePointerUp(e: PointerEvent) {
  const wasDrag = drag !== null;
  drag = null;
  dragging.value = false;
  // 空白处单击（未拖动）→ 关闭
  if (wasDrag && !moved && e.target === stageEl.value) close();
}

function onStageWheel(e: WheelEvent) {
  e.preventDefault();
  zoom(e.deltaY < 0 ? 1.1 : 1 / 1.1);
}

function onDocKeydown(e: KeyboardEvent) {
  if (open.value && e.key === "Escape") close();
}

function onDocPaste(e: ClipboardEvent) {
  if (!open.value) return;
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const blob = item.getAsFile();
      if (!blob) continue;
      const u = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        if (pastedUrl) URL.revokeObjectURL(pastedUrl);
        pastedUrl = u;
        openImg(u, img.naturalWidth, img.naturalHeight);
        notify("📋 已粘贴剪贴板图片");
      };
      img.src = u;
      e.preventDefault();
      break;
    }
  }
}

function onClick(e: MouseEvent) {
  if (open.value && (e.target as HTMLElement).closest(".vz-root")) return;
  const t = e.target as Element;
  // 1) 位图 <img>
  const img = t.closest?.("img") as HTMLImageElement | null;
  if (img && !img.closest(".vz-root")) {
    const src = img.currentSrc || img.src;
    if (!src) return;
    e.preventDefault();
    openImg(src, img.naturalWidth || 800, img.naturalHeight || 600);
    return;
  }
  // 2) Mermaid 渲染的 SVG
  const svg = t.closest?.(".mermaid-svg svg") as SVGSVGElement | null;
  if (svg) {
    e.preventDefault();
    openSvg(svg);
  }
}

onMounted(() => {
  document.addEventListener("click", onClick, true);
  document.addEventListener("keydown", onDocKeydown);
  document.addEventListener("paste", onDocPaste);
});

onBeforeUnmount(() => {
  document.removeEventListener("click", onClick, true);
  document.removeEventListener("keydown", onDocKeydown);
  document.removeEventListener("paste", onDocPaste);
  document.body.style.overflow = "";
});
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="vz-root" role="dialog" aria-modal="true" aria-label="图片查看器">
      <div
        ref="stageEl"
        class="vz-stage"
        :class="{ dragging }"
        @pointerdown="onStagePointerDown"
        @pointermove="onStagePointerMove"
        @pointerup="onStagePointerUp"
        @wheel="onStageWheel"
      >
        <div
          ref="wrapEl"
          class="vz-wrap"
          :style="{
            transform: `translate(calc(-50% + ${tx}px), calc(-50% + ${ty}px))`,
          }"
        >
          <img
            v-if="mode === 'img'"
            :src="url"
            draggable="false"
            class="vz-img"
            :style="{ width: `${Math.round(naturalW * scale)}px` }"
            alt=""
          />
        </div>
      </div>

      <div class="vz-toolbar">
        <button class="vz-btn" title="缩小 (10% 下限)" aria-label="缩小" @click="zoomOut">
          <svg viewBox="0 0 24 24"><path d="M5 12h14" /></svg>
        </button>
        <button class="vz-btn vz-percent" title="点击回到 100%" aria-label="缩放比例，点击复位" @click="resetScale">
          {{ Math.round(scale * 100) }}%
        </button>
        <button class="vz-btn" title="放大 (700% 上限)" aria-label="放大" @click="zoomIn">
          <svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14" /></svg>
        </button>
        <span class="vz-sep"></span>
        <button class="vz-btn" title="复制图片" aria-label="复制图片" @click="copyImage">
          <svg viewBox="0 0 24 24">
            <rect x="9" y="9" width="11" height="11" rx="2" />
            <path d="M5 15V5a2 2 0 0 1 2-2h10" />
          </svg>
        </button>
        <button class="vz-btn" title="关闭 (Esc)" aria-label="关闭" @click="close">
          <svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18" /></svg>
        </button>
      </div>

      <div v-if="toast" class="vz-toast">{{ toast }}</div>
      <div class="vz-hint">滚轮缩放 · 拖拽平移 · Esc 关闭 · Ctrl+V 粘贴图片</div>
    </div>
  </Teleport>
</template>
