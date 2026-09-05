<script setup lang="ts">
import { onMounted, ref } from "vue";

const props = defineProps<{ code: string }>();

const host = ref<HTMLElement>();
const error = ref("");
const uid = "mermaid-" + Math.random().toString(36).slice(2, 10);

function b64decode(s: string): string {
  const bin = atob(s);
  const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
  return new TextDecoder("utf-8").decode(bytes);
}

onMounted(async () => {
  try {
    const mermaid = (await import("mermaid")).default;
    mermaid.initialize({ startOnLoad: false, securityLevel: "loose", theme: "default" });
    const { svg } = await mermaid.render(uid, b64decode(props.code));
    if (host.value) host.value.innerHTML = svg;
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  }
});
</script>

<template>
  <div class="mermaid-host">
    <div v-if="error" class="mermaid-error">
      <strong>Mermaid 渲染失败</strong><br />
      <code>{{ error }}</code>
    </div>
    <div ref="host" v-show="!error" class="mermaid-svg"></div>
  </div>
</template>
