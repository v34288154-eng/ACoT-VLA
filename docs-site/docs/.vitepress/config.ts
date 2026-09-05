import { defineConfig } from "vitepress";
import type MarkdownIt from "markdown-it";

// 部署基础路径：GitHub Pages 项目站点请设 /ACoT-VLA/（在 CI 环境变量中注入），根路径部署用 /
const base = process.env.VITEPRESS_BASE ?? "/";

/**
 * 让 Markdown 中 ```mermaid 代码块渲染为 <Mermaid/> 组件。
 * 将内容做 base64 编码传递，避免引号/换行转义问题；组件在浏览器端渲染(SSR 安全)。
 */
function mermaidFencePlugin(md: MarkdownIt) {
  const defaultFence =
    md.renderer.rules.fence ??
    ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    if (token.info.trim() === "mermaid") {
      const code = Buffer.from(token.content, "utf8").toString("base64");
      return `<Mermaid code="${code}" />`;
    }
    return defaultFence(tokens, idx, options, env, self);
  };
}

export default defineConfig({
  lang: "zh-CN",
  title: "ACoT-VLA",
  description:
    "ACoT-VLA (Action Chain-of-Thought for Vision-Language-Action Models) 使用说明与实现细节文档",
  base,
  cleanUrls: true,
  lastUpdated: true,

  markdown: {
    config: (md) => {
      md.use(mermaidFencePlugin as any);
    },
    image: { lazyLoading: true },
  },

  themeConfig: {
    siteTitle: "ACoT-VLA 文档",
    nav: [
      { text: "首页", link: "/" },
      { text: "总览", link: "/overview" },
      {
        text: "GitHub",
        link: "https://github.com/AgibotTech/ACoT-VLA",
      },
    ],
    sidebar: [
      {
        text: "入门",
        items: [
          { text: "总览 Overview", link: "/overview" },
          { text: "快速上手 Quickstart", link: "/quickstart" },
        ],
      },
      {
        text: "实现细节",
        items: [
          { text: "架构总览 Architecture", link: "/architecture" },
          { text: "数据管线 Data Pipeline", link: "/data" },
          { text: "模型深潜 Model (EAR/IAR)", link: "/model-acot" },
          { text: "训练系统 Training", link: "/training" },
          { text: "配置中心 Config Center", link: "/config-center" },
        ],
      },
      {
        text: "使用与部署",
        items: [
          { text: "推理与服务 Inference", link: "/inference" },
          { text: "评测与竞赛 Evaluation", link: "/evaluation" },
          { text: "真机部署 Real-Robot", link: "/deploy-real" },
        ],
      },
      { text: "附录", items: [{ text: "目录/术语/FAQ", link: "/appendix" }] },
    ],
    outline: { level: [2, 3], label: "本页目录" },
    docFooter: { prev: "上一篇", next: "下一篇" },
    lastUpdated: { text: "最后更新于" },
    search: { provider: "local" },
    socialLinks: [{ icon: "github", link: "https://github.com/AgibotTech/ACoT-VLA" }],
  },
});
