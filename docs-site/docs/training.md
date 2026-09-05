# 训练系统 Training

> 📌 **P0 占位页** —— 正文将由内容路线图 (P2/P3) 基于仓库源码勘察填充；本节先固化"规划内容"供评审。

**规划内容清单：**

- train.py 主循环 / acot_train_step 两步前向
- 两段流匹配损失公式与时间表 Beta 采样
- 优化器(AdamW/SGD)与学习率调度
- FSDP/OSS 分片 mesh 与激活分片约束
- orbax 检查点、assets 资产与权重加载器(预训练合并)
- wandb 日志与实验管理
