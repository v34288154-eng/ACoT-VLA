# 模型深潜 Model (EAR/IAR)

> 📌 **P0 占位页** —— 正文将由内容路线图 (P2/P3) 基于仓库源码勘察填充；本节先固化"规划内容"供评审。

**规划内容清单：**

- ACOTConfig 字段逐项解释(变体/维度/双horizon/开关)
- PaliGemma 主干: SigLIP 图像编码 + Gemma LLM 前缀
- 双专家机制: gemma.Module 三配置共享深度扫掠 + adaRMS
- EAR: Coarse Expert 流匹配与粗轨迹生成
- IAR: 三种 extractor(Downsample/LearnableQuery/Pooling) + 交叉注意力
- 推理融合(UnifiedAttention/MLP) 与注意力掩码/AR 语义
- 冻结策略 get_freeze_filter 与 LoRA 支持
- 论文符号 ↔ 代码命名对照表
