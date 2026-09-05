# 数据管线 Data Pipeline

> 📌 **P0 占位页** —— 正文将由内容路线图 (P2/P3) 基于仓库源码勘察填充；本节先固化"规划内容"供评审。

**规划内容清单：**

- LeRobot 数据集格式约定(state/action/粗动作轨迹字段)
- convert_libero/aloha/droid 转换脚本说明
- 归一化统计量: RunningStats/分位数、norm_stats 资产与加载
- transforms 流水线逐级讲解(含 ACOT 专用变换)
- FrameSampler 采样策略 与 DataLoaderACOT 产出
- 粗/细动作窗口语义: 步长 shift(2,1) 与 horizon 换算
