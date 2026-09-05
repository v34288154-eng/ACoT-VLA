# 命名配置注册表（自动生成）

> 来源：`src/openpi/training/config.py` 的 `_CONFIGS` 列表；生成时间：2026-09-05 17:16；基线 commit：`c323377`。
> 统计：共 25 个命名配置（按模型类粗略分类：ACOT×8 · Pi0×17）。运行时以 `get_config(name)` 查询，CLI 经 tyro 覆盖字段。
> 由 `python tools/config_registry.py --repo <repo> --commit <sha>` 生成，请勿手改。

| 配置名 | 行号 | 模型 | 数据工厂 | 模型关键覆盖 |
|---|---|---|---|---|
| `pi0_aloha` | 1245 | pi0.Pi0Config | LeRobotAlohaDataConfig |  |
| `pi05_aloha` | 1253 | pi0.Pi0Config | LeRobotAlohaDataConfig | pi05=True |
| `pi0_aloha_towel` | 1261 | pi0.Pi0Config | LeRobotAlohaDataConfig |  |
| `pi0_aloha_tupperware` | 1270 | pi0.Pi0Config | LeRobotAlohaDataConfig |  |
| `pi0_droid` | 1282 | pi0.Pi0Config | SimpleDataConfig | action_horizon=10 |
| `pi0_fast_droid` | 1296 | pi0_fast.Pi0FASTConfig | SimpleDataConfig | action_dim=8, action_horizon=10 |
| `pi05_droid` | 1310 | pi0.Pi0Config | SimpleDataConfig | action_horizon=15, pi05=True |
| `pi0_libero` | 1332 | pi0.Pi0Config | LeRobotLiberoDataConfig |  |
| `pi0_libero_low_mem_finetune` | 1359 | pi0.Pi0Config | LeRobotLiberoDataConfig | paligemma_variant=gemma_2b_lora, action_expert_variant=gemma_300m_lora |
| `pi0_fast_libero` | 1380 | pi0_fast.Pi0FASTConfig | LeRobotLiberoDataConfig | action_dim=7, action_horizon=10, max_token_len=180 |
| `pi0_fast_libero_low_mem_finetune` | 1402 | pi0_fast.Pi0FASTConfig | LeRobotLiberoDataConfig | action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant=gemma_2b_lora |
| `pi05_libero` | 1424 | pi0.Pi0Config | LeRobotLiberoDataConfig | pi05=True, action_horizon=10, discrete_state_input=False |
| `pi0_aloha_pen_uncap` | 1451 | pi0.Pi0Config | LeRobotAlohaDataConfig |  |
| `pi05_aloha_pen_uncap` | 1480 | pi0.Pi0Config | LeRobotAlohaDataConfig | pi05=True |
| `pi0_fast_full_droid_finetune` | 1515 | pi0_fast.Pi0FASTConfig | RLDSDroidDataConfig | action_dim=8, action_horizon=16, max_token_len=180 |
| `pi05_droid_finetune` | 1545 | pi0.Pi0Config | LeRobotDROIDDataConfig | pi05=True, action_dim=32, action_horizon=16 |
| `pi0_aloha_sim` | 1572 | pi0.Pi0Config | LeRobotAlohaDataConfig |  |
| `acot_libero_action_cot_explicit_implicit_co_fusion` | 1584 | acot_vla.ACOTConfig | LeRobotACOTLiberoDataConfig | coarse_action_horizon=15, action_horizon=10, pi05=True, discrete_state_input=False, coarse_action_expert_va... |
| `acot_vlabench_action_cot_explicit_implicit_co_fusion` | 1611 | acot_vla.ACOTConfig | LeRobotACOTVLABenchDataConfig | action_dim=7, coarse_action_horizon=15, action_horizon=10, pi05=True, discrete_state_input=True, coarse_act... |
| `acot_libero_plus_action_cot_explicit_implicit_co_fusion` | 1649 | acot_vla.ACOTConfig | LeRobotACOTLiberoPlusDataConfig | coarse_action_horizon=15, action_horizon=10, pi05=True, discrete_state_input=False, coarse_action_expert_va... |
| `acot_go1_openset_pick_action_cot_explicit_implicit_co_fusion` | 1676 | acot_vla.ACOTConfig | LerobotACOTGo1DataConfig | coarse_action_horizon=30, action_horizon=30, pi05=True, discrete_state_input=False, action_expert_variant=g... |
| `acot_go1_wipe_stain_action_cot_explicit_implicit_co_fusion` | 1719 | acot_vla.ACOTConfig | LerobotACOTGo1DataConfig | coarse_action_horizon=30, action_horizon=30, pi05=True, discrete_state_input=True, action_expert_variant=ge... |
| `acot_go1_pourwater_action_cot_explicit_implicit_co_fusion` | 1745 | acot_vla.ACOTConfig | LerobotACOTGo1DataConfig | coarse_action_horizon=30, action_horizon=30, pi05=True, discrete_state_input=True, action_expert_variant=ge... |
| `acot_agilex_openset_pick_action_cot_explicit_implicit_co_...` | 1772 | acot_vla.ACOTConfig | LerobotACOTAgilexDataConfig | coarse_action_horizon=30, action_horizon=30, pi05=True, discrete_state_input=False, action_expert_variant=g... |
| `acot_icra_simulation_challenge_reasoning_to_action` | 1816 | acot_vla.ACOTConfig | LerobotACOTGo2DataConfig | coarse_action_horizon=30, action_horizon=30, paligemma_variant=gemma_2b_lora, adopt_explicit_action_reasone... |
