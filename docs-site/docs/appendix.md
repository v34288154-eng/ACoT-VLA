---
title: 附录 Appendix
---

> 核验于源码基线 commit cb9d195。

# 附录 Appendix

本页是文档站的"收尾页"，集中提供五类导航与速查内容：**站点地图**（全站 11 个栏目页 + 3 个自动生成快照页的入口）、三个自动生成页的**使用方法**、**术语表**（34 条，按模型与推理 / 数据与训练 / 工程与部署三组）、**FAQ**（14 条常见坑）与**本仓库相对 OpenPI 的差异对照表**，末尾附许可与致谢。

正文涉及代码的行号均已对照源码核实；三个自动生成快照页（`/generated/*`）头部各自标注了生成时间与生成基线 commit，若与本页引用的 `cb9d195` 不一致，以快照页自身标注为准。

## 站点地图

站点共 11 个栏目页（对应侧边栏分组：入门 2 · 实现细节 5 · 使用与部署 3 · 附录 1），另有首页 `/`（landing）与 3 个自动生成页。栏目页导航如下：

| 页面 | 一句话用途 |
| --- | --- |
| [总览 Overview](/overview) | 项目定位、三大核心组件（EAR/IAR/ACoT）、LIBERO 等基准成绩、与 OpenPI 的关系与仓库一览 |
| [快速上手 Quickstart](/quickstart) | 安装 → 数据 → norm stats → 训练 → 策略服务 → 验证的最小闭环，命令逐字取自仓库 |
| [架构总览 Architecture](/architecture) | 系统分层总述与 Mermaid 主架构图（数据、模型、训练、推理部署一图流） |
| [数据管线 Data Pipeline](/data) | LeRobot 数据格式、各数据集转换脚本、归一化统计量与 transforms（含 ACoT 变换） |
| [模型深潜 Model (EAR/IAR)](/model-acot) | ACOTConfig 字段、EAR/IAR/双动作专家/融合/冻结策略与 LoRA 的实现细节 |
| [训练系统 Training](/training) | train 主循环、两段流匹配损失、FSDP/OSS 分片、orbax 检查点与资产 |
| [配置中心 Config Center](/config-center) | 全部命名配置总表、ACoT 家族逐项释义、tyro CLI 用法 |
| [推理与服务 Inference](/inference) | sample_actions 两阶段采样、serve_policy WebSocket 协议、openpi-client |
| [评测与竞赛 Evaluation](/evaluation) | LIBERO/LIBERO-Plus/VLABench 复现、ICRA2026 竞赛提交闭环 |
| [真机部署 Real-Robot](/deploy-real) | ALOHA real/sim、DROID/UR5/go1/AgileX 案例与 server/client 接线模式 |
| [附录 Appendix](/appendix) | 本页：站点地图、快照入口、术语表、FAQ、OpenPI 差异对照、许可与致谢 |

自动生成页（存放在 `docs/generated/`，由脚本自动导出，**请勿手改**）：

| 页面 | 一句话用途 |
| --- | --- |
| [目录树快照](/generated/repo-tree) | 仓库完整目录树（相对路径 + 文件大小），快速定位脚本/示例/模块文件 |
| [模块地图](/generated/module-map) | 公开符号索引：类/函数/方法 → 文件:行号 |
| [配置注册表](/generated/config-registry) | `_CONFIGS` 全部命名配置清单（配置名 → 行号 → 模型 → 数据工厂 → 关键覆盖） |

## 目录树快照

完整目录快照见 [目录树快照](/generated/repo-tree)：由 `docs-site/tools/scan_repo.py` 对仓库根目录自动扫描生成（逐文件列出相对路径与大小），页首标注生成时间与基线 commit。适合在通读前快速建立"仓库里有什么"的全局印象，例如查找某个示例/脚本/文档的准确路径。

## 符号与模块索引

符号级索引见 [模块地图](/generated/module-map)：由 `docs-site/tools/module_map.py` 自动生成，覆盖 `src/openpi`、`packages/openpi-client`、`scripts`、`examples` 下的公开符号，按**源文件分组**、以 `符号名 (行 N)` 形式给出定位。用法：想知道"某个类/函数定义在哪个文件的哪一行"，在页内搜索符号名即可，再配合编辑器跳转核对。

## 配置注册表

全部命名配置见 [配置注册表](/generated/config-registry)：由 `docs-site/tools/config_registry.py` 从 `src/openpi/training/config.py` 的 `_CONFIGS` 列表导出。统计口径以该页为准：共 **25 个命名配置，其中 ACoT 家族 8 个**（其余 17 个为 Pi0/Pi0FAST 系，沿用上游命名）；训练/推理脚本以 `--config-name <名字>` 取用，运行时经 tyro 覆盖字段。想给自有数据集新增一个命名配置的步骤见 /config-center。

## 术语表

以下 34 条按三组列出；仓库专有词以源码为准并标注了关键出处行号，通用词给领域内通行解释。行号类引用可在 [模块地图](/generated/module-map) 反查。

### 模型与推理（12 条）

| 术语 | 一句话解释 |
| --- | --- |
| **ACoT（Action Chain-of-Thought，动作链式思考）** | 论文提出的新范式：把 VLA 的"斟酌"直接搬进动作空间，由 EAR 与 IAR 协同产出一串结构化动作意图，作为 grounded 长程策略学习的中间推理链。 |
| **EAR（Explicit Action Reasoner，显式动作推理器）** | 轻量 Transformer（配合 Coarse Expert），显式合成**粗粒度运动轨迹**，为精细动作提供直接运动线索；开关为 `adopt_explicit_action_reasoner`（`src/openpi/models/acot_vla.py:282`）。 |
| **IAR（Implicit Action Reasoner，隐式动作推理器）** | 用交叉注意力从 VLM 主干内部表征抽取**隐式动作先验**；开关为 `adopt_implicit_action_reasoner`（`src/openpi/models/acot_vla.py:283`），隐式抽取器三选一：LearnableQuery / AttentionPooling / Downsample（`src/openpi/models/acot_vla.py:285-287`）。 |
| **Coarse/Action Expert（双动作专家）** | 两个 Gemma-300M 专家头（默认 `coarse_action_expert_variant` 与 `action_expert_variant` 均为 `"gemma_300m"`，`src/openpi/models/acot_vla.py:270-271`）：EAR 侧的 Coarse Expert 出粗轨迹，Action Expert 出最终动作块。 |
| **coarse / action horizon（粗/细动作视界）** | 粗轨迹与最终动作块各自覆盖的时间步数：模型默认 `coarse_action_horizon=50`、`action_horizon=30`（`src/openpi/models/acot_vla.py:275-276`）；命名配置常取 15/10（LIBERO 系）或 30/30（go1/ICRA 等长程任务）。 |
| **Action Chunk（动作块）** | 一次模型推理输出的连续动作序列（形状 `[action_horizon, action_dim]`），由执行侧按控制节拍逐帧消费（见 ActionChunkBroker 一条）。 |
| **Flow Matching（流匹配）** | 扩散式生成范式（与 π0 同源）：训练阶段回归"噪声 → 数据"的速度场，推理时从纯噪声做数值积分采样；Coarse 与 Action 两条专家路径各有一路流匹配。 |
| **prefix KV Cache（前缀 KV 缓存）** | PaliGemma 主干对"图像 + 指令"前缀只做一次前向并缓存 KV；EAR/IAR 与动作专家在缓存之上继续计算，避免每次推理重复编码前缀。 |
| **Frozen（冻结）** | 训练时冻结部分权重、不参与更新；README/成绩表里的 "Frozen" 指冻结 LLM 主干。冻结哪些路径由 `get_freeze_filter`（`src/openpi/models/acot_vla.py:333`）决定（可分别冻结 vision / LLM / 双专家），LoRA 场景下冻结路径放行 LoRA 参数。 |
| **LoRA** | 低秩适配微调：只训练注入的低秩旁路以降低显存与显存开销；实现见 `src/openpi/models/lora.py`，配置变体如 `paligemma_variant="gemma_2b_lora"`（`pi0_libero_low_mem_finetune` 等配置使用）。 |
| **pi05（π0.5 设定）** | `ACOTConfig.pi05`（默认 True，`src/openpi/models/acot_vla.py:278`）开启 π0.5 风格设定：决定 `max_token_len`（200 vs 48，`src/openpi/models/acot_vla.py:290-291`）与模型类型（`ACOT_VLA_PI05` vs `ACOT_VLA_PI0`，`src/openpi/models/acot_vla.py:297-301`），并默认联动离散状态输入。 |
| **离散状态输入（discrete_state_input）** | 默认跟随 `pi05`（`src/openpi/models/acot_vla.py:292-293`）：开启时把本体状态作为**离散语言输入**的一部分交给分词器（`src/openpi/models/tokenizer.py:21`）；关闭时状态不进语言模型。 |

### 数据与训练（11 条）

| 术语 | 一句话解释 |
| --- | --- |
| **LeRobot** | HuggingFace 生态的数据集格式/库：统一以 HF dataset 存放 `state`、`actions` 等字段；仓库按固定 git rev 依赖 lerobot（pyproject.toml），LIBERO/DROID/真机数据都先转成 LeRobot 再使用。 |
| **RLDS** | 部分公开基准的原始数据格式（如 LIBERO 原始集来自 `openvla/modified_libero_rlds`、DROID 数据集）；仓库以 `convert_*_data_to_lerobot.py` 脚本转成 LeRobot 格式。 |
| **norm stats（归一化统计量）** | 训练前对 `state`、`actions`、`coarse_actions` 等键滚动统计出的均值/方差（可含分位数）资产；由 `scripts/compute_norm_stats.py` 生成 `norm_stats.json`，训练数据管线启动时加载并做归一化（生成/加载位置见 /quickstart「3. 计算归一化统计量」）。 |
| **assets（资产目录）** | 模型运行所需旁路资产的存放处，默认 `./assets/<配置名>`：归一化统计量与预训练/合并权重都从资产目录按 `asset_id`（默认等于数据集 repo_id）加载；缺失只打日志跳过。 |
| **RepackTransform（重打包变换）** | 按"新键 → 原始嵌套路径"的映射把原始样本重排成模型输入字段（相机图、state、actions 等），见 `src/openpi/transforms.py:82`。 |
| **ACOT 增量/绝对动作变换（ACOTDeltaActions / ACOTAbsoluteActions）** | 在绝对与增量动作空间之间互转，并同时作用于粗/细两个动作键 `coarse_actions` 与 `actions`（`src/openpi/transforms.py:258` 与 `:278`）。 |
| **shift 窗口（joint_action_shifts）** | 切分动作窗口时的滑动步长，默认 `(2, 1)`：粗动作窗口每步走 2 帧、细动作窗口每步走 1 帧（`src/openpi/training/config.py:454`）；与两个 horizon 共同决定动作块长度（`src/openpi/training/data_loader.py:179-180`）。 |
| **FrameSampler** | 训练数据采样器：只从解析出的合法时间区间（如子任务区间）内取帧并打乱（`src/openpi/training/sampler.py:66`）。 |
| **DataLoaderACOTImpl** | ACoT 模型专用的数据装载实现：每批产出 `(Observation, actions, coarse_actions)` 三元组，把粗动作一并喂给训练（`src/openpi/training/data_loader.py:599`）；普通模型走 `DataLoaderImpl`（同文件 `:587`）。 |
| **两段流匹配损失** | ACoT 的训练目标：Coarse Expert 与 Action Expert 各算一段流匹配回归损失后相加（形式见 /architecture 主架构图 ④ 训练子系统）。 |
| **checkpoint（检查点）** | 训练按 `save_interval` 以 orbax 目录树形式写入 `./checkpoints/<CONFIG_NAME>/<EXP_NAME>/<step>`（含参数与优化器状态），推理/评测按该路径加载。 |

### 工程与部署（11 条）

| 术语 | 一句话解释 |
| --- | --- |
| **OpenPI（上游框架）** | 本仓库构建其上的开源 π0 训练/推理框架；工程布局、LeRobot/RLDS 数据、流匹配训练与 WebSocket 策略服务等均继承自它（见 README 致谢段与下文差异对照表）。 |
| **orbax** | JAX 生态的检查点库（依赖 `orbax-checkpoint==0.11.13`），负责保存/恢复模型参数与优化器状态树。 |
| **FSDP / OSS** | 训练分片策略：FSDP 前向/反向时对参数分片，OSS 对优化器状态分片，配合训练 mesh 跨设备分布（见 /architecture ④）。 |
| **tyro** | 基于 dataclass 配置自动生成 CLI 的库：训练/推理脚本用 `--config-name <名字>` 选命名配置，并以 tyro 语法覆盖嵌套字段。 |
| **uv** | Python 包与虚拟环境管理器：README 的安装（`uv sync` + `uv pip install -e .`）与仓库脚本（`uv run ...`）统一经它执行。 |
| **policy server（策略服务器）** | 把已训练策略封装成 WebSocket 服务（`scripts/serve_policy.py`，常以 `server.sh` 启动），接收客户端观测并回传动作。 |
| **ActionChunkBroker** | openpi-client 中的包装策略（`packages/openpi-client/src/openpi_client/action_chunk_broker.py:10`）：首次请求触发整块推理并缓存结果，之后每次 infer 只取块内单步，块耗尽才再次推理——让执行循环按节拍逐帧消费动作块。 |
| **openpi-client** | 独立客户端包（`packages/openpi-client`）：内含 `WebsocketClientPolicy`（`packages/openpi-client/src/openpi_client/websocket_client_policy.py:12`）、ActionChunkBroker、runtime 等；评测环境与真机都通过它连接策略服务。 |
| **命名配置（named config）** | 注册于 `src/openpi/training/config.py` 的 `_CONFIGS`、可按名取用的训练配置：共 25 个（ACoT×8 + Pi0×17），完整清单见 [配置注册表](/generated/config-registry)。 |
| **ICRA2026 Reasoning2Action baseline** | 本仓库作为 AgiBot World Challenge @ ICRA 2026（Reasoning to Action 赛道）官方 baseline：入口配置 `acot_icra_simulation_challenge_reasoning_to_action`（`src/openpi/training/config.py:1817`，Go2、30/30 双 horizon、EAR+IAR 全开），配套服务脚本固定 `--env G2SIM`。 |
| **third_party 子模块** | 以 git submodule 挂载的第三方依赖（`third_party/aloha`、`third_party/libero`，见仓库 `.gitmodules`）；安装需先 `git submodule update --init --recursive`，缺失会导致依赖它的示例（如 LIBERO）直接 import 失败。 |

## FAQ

按主题分组的 14 条常见问题；需要命令细节的答案统一指向对应页面，正文不复述命令。

### 环境与安装

- **问：装完环境后 `import openpi` 仍报错？**
  最常见原因是只跑了 `uv sync` 而没有执行 `uv pip install -e .`，顶层 `openpi` 包未装入环境；两条 uv 命令缺一不可，命令见 /quickstart「1. 环境安装」。

- **问：为什么必须初始化 third_party 子模块？**
  LIBERO 等示例的数据转换/评测代码依赖子模块（aloha/libero），跳过 `git submodule update` 会在 import 阶段直接失败；见 /quickstart「1. 环境安装」与术语表「third_party 子模块」。

- **问：一定要 NVIDIA GPU 吗？**
  是。训练/推理基于 `jax[cuda12]`（依赖 CUDA），官方流程假设 NVIDIA GPU；硬件要求见 /quickstart「0. 前置条件」。

### 数据

- **问：训练时数据集报错 / repo_id 不对怎么办？**
  ACoT 系命名配置的默认 `repo_id` 多为作者本机路径（如 `/mnt/public/...`），`pi0_libero` 默认指向 HF 公共集——把配置里的 `repo_id` 换成你自己的数据集（本地路径或 HF repo）即可，见 /quickstart「常见坑 1」与 /data。

- **问：忘了先算归一化统计量会怎样？**
  数据管线只会打一行日志然后跳过，模型会在**未归一化**的数据上训练、效果异常；请先为所用命名配置生成 norm stats 并确认资产目录位置，见 /quickstart「3. 计算归一化统计量」与「常见坑 7」。

- **问：想把自有数据喂进来，从哪入手？**
  先按 LeRobot 格式整理数据集，再新增/复用命名配置并指向该数据集；格式约定与转换脚本见 /data，新增配置步骤见 /config-center「为自有数据集新增配置」。

### 配置与模型

- **问：`acot_*` 与 `pi0_*` 配置有什么区别？**
  `pi0_*` 是继承自 OpenPI 的 π0/π0.5 单动作专家系；`acot_*` 是新增的 ACoT 模型族（双动作专家 + EAR/IAR + 双 horizon）。完整清单与逐项释义见 /config-center 与 [配置注册表](/generated/config-registry)。

- **问：`pi05` / `discrete_state_input` 这些开关需要自己改吗？**
  一般不用：`discrete_state_input` 默认跟随 `pi05`（见术语表「pi05」「离散状态输入」），8 个 `acot_*` 注册配置均已按任务显式设定；想按任务调整时先读 /model-acot 的字段解释再改。

- **问：显存不够 / 想快速冒烟怎么办？**
  首选低显存 LoRA 配置（如 `pi0_libero_low_mem_finetune`），或开 `DEBUG_MODE` 冒烟运行（自动退化为 batch=1、缩短保存间隔）；见 /quickstart「4. 训练」与「常见坑 5」，完整调参见 /training。

### 训练与日志

- **问：检查点存在哪里、如何续训/加载？**
  按 `./checkpoints/<CONFIG_NAME>/<EXP_NAME>/<step>` 组织；最小闭环里的目录约定见 /quickstart「4. 训练」，orbax 保存/加载与权重合并细节见 /training。

- **问：wandb 必须登录才能训练吗？**
  不必。日志默认开启但 `train.sh` 预设了离线模式（`WANDB_MODE=offline`），不登录也能本地记录；要看在线曲线先 `wandb login`，见 /quickstart「常见坑 6」。

### 推理与部署

- **问：`server.sh` 起的是哪个模型？**
  脚本内部固定 `--env G2SIM` 并指向 ICRA baseline 的检查点路径；要服务自己训的模型，请直接调 `serve_policy.py` 显式指定 env 与 checkpoint，见 /quickstart「5. 启动策略服务」。

- **问：`simple_client` 能当评测用吗？**
  不能直接当成绩评测：它用随机观测验证"客户端 ↔ 策略服务"连通与推理吞吐，不代表任务成功率；真实仿真评测见 /evaluation。

- **问：真机部署和仿真评测的接线方式有何区别？**
  两者都走"policy server + openpi-client"模式；真机多出机器人/相机驱动、动作空间与标定等环节，见 /deploy-real；仿真环境侧见 /evaluation。

## 与 OpenPI 的差异对照表

下表只列**仓库内可核实**的差异（对照源码与生成快照验证）。"继承"指继承自 OpenPI 的工程布局（`src/openpi`、`packages/openpi-client`、LeRobot/RLDS 数据、流匹配训练、WebSocket 策略服务等），"新增"为本仓库相对上游的扩展；完整目录见 [目录树快照](/generated/repo-tree)。

| 对比项 | ACoT-VLA（本仓库） | 关系 |
| --- | --- | --- |
| 工程布局与模块划分 | 沿用 `src/openpi`（模型/训练）+ `packages/openpi-client`（客户端）目录布局，examples/ 示例与 third_party 子模块机制同上游一致 | 继承（README 致谢确认构建于 OpenPI 之上） |
| 数据与训练/服务范式 | LeRobot/RLDS 数据管线、归一化资产、流匹配训练、WebSocket 策略服务 + openpi-client | 继承，并叠加下方 ACoT 专用扩展 |
| 模型族 | 新增 `src/openpi/models/acot_vla.py`：`ACOTConfig`（`:267`）/ `ACOT_VLA`（`:376`），含 EAR/IAR 开关、三种隐式抽取器（LearnableQuery `:35`、AttentionPooling `:105`、Downsample `:156`）与融合模块 `UnifiedAttentionModule`（`:242`） | 新增 |
| 训练配置注册 | `_CONFIGS` 新增 8 个 `acot_*` 命名配置（全表 25 = ACoT×8 + Pi0×17），Pi0 系沿用上游命名 | 新增 |
| ACOT 数据工厂族 | `config.py` 中新增：`LeRobotACOTLiberoDataConfig`（`:666`）、`LeRobotACOTLiberoPlusDataConfig`（`:713`）、`LeRobotACOTVLABenchDataConfig`（`:451`）、`LerobotACOTGo1DataConfig`（`:504`）、`LerobotACOTGo2DataConfig`（`:583`）、`LerobotACOTAgilexDataConfig`（`:928`）、`LerobotACOTARXDataConfig`（`:1074`），均带 `joint_action_shifts=(2,1)` 双窗口 | 新增 |
| 数据装载 | 新增 `DataLoaderACOTImpl`（`src/openpi/training/data_loader.py:599`）：每批产出含 `coarse_actions` 的三元组；普通模型沿用 `DataLoaderImpl`（同文件 `:587`） | 新增 |
| 动作表示与数据变换 | 新增粗动作通道 `CoarseActions`（`src/openpi/models/model.py:152`），以及同时作用于 `coarse_actions`/`actions` 两键的 `ACOTDeltaActions`/`ACOTAbsoluteActions`（`src/openpi/transforms.py:258`/`:278`） | 新增 |
| 归一化统计 | `scripts/compute_norm_stats.py` 的统计键扩展到 `coarse_actions`（对不产出该键的配置整批跳过） | 继承 + 扩展 |
| 权重加载 | 新增 `ACOTCheckpointWeightLoader`（`src/openpi/training/weight_loaders.py:57`）等 ACoT 预训练权重合并/加载路径 | 新增 |
| 竞赛 baseline | 官方 ICRA2026 baseline 配置 `acot_icra_simulation_challenge_reasoning_to_action`（`src/openpi/training/config.py:1817`，Go2、30/30 双 horizon、EAR+IAR 全开），服务端固定 `--env G2SIM` | 新增（本仓库即该赛道官方 baseline） |
| 机器人/真机平台 | 在继承的 examples 布局上扩展 go1/go2/Agilex/ARX 等平台的数据与策略支持（数据工厂见配置注册表），examples 内另有 aloha_real、ur5 等真机示例 | 新增扩展 |

## 许可与致谢

本仓库在 README 顶部标注 **MIT**（代码）与 **CC BY 4.0**（论文与资源）双许可徽章，并明确致谢：*"This repo is built upon the OpenPI framework"*——文档与引用格式（bibtex）以 [仓库 README](https://github.com/AgibotTech/ACoT-VLA) 全文为准。

## 相关页面

- [总览 Overview](/overview) —— 项目定位、核心组件与性能速览
- [快速上手 Quickstart](/quickstart) —— 最小闭环：安装 → 数据 → 训练 → 服务 → 验证
- [架构总览 Architecture](/architecture) —— 系统分层与主架构图
- [数据管线 Data Pipeline](/data) —— LeRobot 格式、转换与归一化
- [模型深潜 Model (EAR/IAR)](/model-acot) —— ACoTConfig 与 EAR/IAR/融合实现
- [训练系统 Training](/training) —— 训练循环、两段损失与分片
- [配置中心 Config Center](/config-center) —— 命名配置注册表与调参
- [推理与服务 Inference](/inference) —— 策略服务器与 rollout
- [评测与竞赛 Evaluation](/evaluation) —— LIBERO/VLABench/ICRA 评测闭环
- [真机部署 Real-Robot](/deploy-real) —— 真机环境与 server/client 部署
- 生成页：[目录树快照](/generated/repo-tree) · [模块地图](/generated/module-map) · [配置注册表](/generated/config-registry)
