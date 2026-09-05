# 架构总览 Architecture

> 📌 **P0 已内置主架构图(渲染验证用)**；配套的分层说明、模块地图表格与推理/训练局部图将由 P2 阶段填充。

**规划内容清单：**

- 系统分层总述：数据与评测环境 / 离线数据管线 / 模型 / 训练 / 推理部署
- 模块地图表(路径 → 职责 → 关键符号)
- 推理链路时序图(sequenceDiagram)
- 训练流程图(两段流匹配损失)

## 主架构图（Mermaid）

```mermaid
flowchart TB
    classDef data  fill:#E8F1FB,stroke:#2F5B93,color:#111
    classDef pipe  fill:#FFF6CC,stroke:#A17B00,color:#111
    classDef model fill:#E4F0DE,stroke:#3F6B2C,color:#111
    classDef train fill:#FDEADA,stroke:#B4501E,color:#111
    classDef dep   fill:#EEE9FA,stroke:#4E3F8F,color:#111
    classDef evn   fill:#FBE3EA,stroke:#9E3155,color:#111

    subgraph G1["① 数据与评测环境"]
        D1["原始数据<br/>LIBERO · LIBERO-Plus · VLABench ·<br/>ALOHA · DROID · UR5 · go1 · AgileX"]:::data
        E1["评测与真机环境<br/>examples/*/main.py<br/>Mujoco 仿真 / 真机控制器"]:::evn
    end

    subgraph G2["② 离线数据管线 (src/openpi)"]
        T1["convert_*_data_to_lerobot.py<br/>→ LeRobot(HuggingFace) 数据集"]:::pipe
        T3["compute_norm_stats.py<br/>→ 归一化统计量资产"]:::pipe
        T2["transforms.py 数据变换<br/>Repack · Resize · ACOTDelta/AbsoluteActions<br/>TokenizePrompt · PadStatesAndActions<br/>粗/细动作窗口切分(shift 2/1)"]:::pipe
    end

    subgraph G3["③ ACoT-VLA 模型 (models/acot_vla.py)"]
        M1["PaliGemma 主干(前缀 Prefix)<br/>SigLIP ViT 图像 tokens + Gemma-2B LLM<br/>→ prefix 前向 → KV Cache"]:::model
        M2["IAR 隐式动作推理<br/>KV Cache 逐层特征 → extractor<br/>(Downsample/LearnableQuery/Pooling)<br/>→ z_im → 交叉注意力 s_im"]:::model
        M3["EAR 显式动作推理(Coarse Expert)<br/>Gemma-300M 流匹配<br/>→ 长时域粗轨迹 z_ex"]:::model
        M4["动作推理融合<br/>UnifiedAttention / MLP<br/>s_ex + s_im → 增强 token"]:::model
        M5["Action Expert 精细动作头<br/>Gemma-300M 流匹配<br/>→ 动作块 (horizon 10~30)"]:::model
    end

    subgraph G4["④ 训练 (training/ + scripts/train.py)"]
        R1["ACoT 两段流匹配损失<br/>‖u_coarse−v_coarse‖² + ‖u_expert−v_expert‖²"]:::train
        R2["优化器 · FSDP/OSS 分片 mesh<br/>orbax 检查点 + assets<br/>wandb 实验日志"]:::train
    end

    subgraph G5["⑤ 推理与部署"]
        S1["sample_actions 两阶段采样<br/>纯噪声 →(EAR 流)→ 粗轨迹<br/>→(Expert 流+推理增强)→ 动作块"]:::model
        S2["serve_policy.py<br/>Websocket 策略服务 (server.sh)"]:::dep
        S3["openpi-client<br/>WebsocketClientPolicy<br/>ActionChunkBroker"]:::dep
    end

    D1 --> T1 --> T2 --> M1
    T3 --> T2
    M1 --> M2
    M2 --> M4
    M1 --> M3
    M3 --> M4
    M4 --> M5
    M1 -. prefix KV .-> M2
    M2 --> R1
    M3 --> R1
    M5 --> R1
    R1 --> R2
    R2 -. 检查点/权重 .-> S1
    S1 --> S2 --> S3
    E1 --> S3
    S3 -. 执行动作/回传观测 .-> E1
```
