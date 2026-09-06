# 仓库目录快照（自动生成）

> 生成时间：2026-09-06 10:38；基线 commit：`592eb4e`。
> 由 `python tools/scan_repo.py --repo <repo> --commit <sha>` 生成，请勿手改。

```text
.
├── .github/
│   └── workflows/
│       ├── docs.yml   (1.3 KB)
│       ├── pre-commit.yml   (0.3 KB)
│       └── test.yml   (0.7 KB)
├── docs/
│   ├── docker.md   (2.0 KB)
│   ├── framework.png   (270.4 KB)
│   ├── norm_stats.md   (5.2 KB)
│   └── remote_inference.md   (3.6 KB)
├── docs-site/
│   ├── docs/
│   │   ├── generated/
│   │   │   ├── config-registry.md   (4.1 KB)
│   │   │   ├── module-map.md   (25.2 KB)
│   │   │   └── repo-tree.md   (8.4 KB)
│   │   ├── public/
│   │   │   └── figures/
│   │   │       └── framework.png   (270.4 KB)
│   │   ├── appendix.md   (21.2 KB)
│   │   ├── architecture.md   (15.1 KB)
│   │   ├── config-center.md   (17.9 KB)
│   │   ├── data.md   (16.4 KB)
│   │   ├── deploy-real.md   (15.8 KB)
│   │   ├── evaluation.md   (11.0 KB)
│   │   ├── index.md   (3.0 KB)
│   │   ├── inference.md   (13.4 KB)
│   │   ├── model-acot.md   (24.8 KB)
│   │   ├── overview.md   (10.4 KB)
│   │   ├── quickstart.md   (12.1 KB)
│   │   └── training.md   (19.7 KB)
│   ├── tools/
│   │   ├── check_anchors.py   (4.6 KB)
│   │   ├── check_links.mjs   (2.0 KB)
│   │   ├── check_mermaid.mjs   (1.9 KB)
│   │   ├── config_registry.py   (5.2 KB)
│   │   ├── module_map.py   (3.9 KB)
│   │   └── scan_repo.py   (2.8 KB)
│   ├── .gitignore   (0.1 KB)
│   ├── package-lock.json   (150.3 KB)
│   └── package.json   (0.5 KB)
├── examples/
│   ├── aloha_real/
│   │   ├── compose.yml   (1.5 KB)
│   │   ├── constants.py   (3.3 KB)
│   │   ├── convert_aloha_data_to_lerobot.py   (7.6 KB)
│   │   ├── Dockerfile   (2.7 KB)
│   │   ├── env.py   (1.8 KB)
│   │   ├── main.py   (1.4 KB)
│   │   ├── README.md   (6.4 KB)
│   │   ├── real_env.py   (8.6 KB)
│   │   ├── requirements.in   (0.2 KB)
│   │   ├── requirements.txt   (3.4 KB)
│   │   ├── robot_utils.py   (9.9 KB)
│   │   └── video_display.py   (1.0 KB)
│   ├── aloha_sim/
│   │   ├── compose.yml   (1.0 KB)
│   │   ├── Dockerfile   (1.5 KB)
│   │   ├── env.py   (2.0 KB)
│   │   ├── main.py   (1.4 KB)
│   │   ├── README.md   (0.8 KB)
│   │   ├── requirements.in   (0.1 KB)
│   │   ├── requirements.txt   (2.7 KB)
│   │   └── saver.py   (1.3 KB)
│   ├── droid/
│   │   ├── convert_droid_data_to_lerobot.py   (17.2 KB)
│   │   ├── main.py   (9.8 KB)
│   │   ├── README.md   (4.8 KB)
│   │   └── README_train.md   (5.3 KB)
│   ├── libero/
│   │   ├── compose.yml   (1.3 KB)
│   │   ├── convert_libero_data_to_lerobot.py   (3.9 KB)
│   │   ├── Dockerfile   (2.4 KB)
│   │   ├── main.py   (10.0 KB)
│   │   ├── README.md   (2.1 KB)
│   │   ├── requirements.in   (0.2 KB)
│   │   └── requirements.txt   (2.9 KB)
│   ├── simple_client/
│   │   ├── compose.yml   (1.0 KB)
│   │   ├── Dockerfile   (1.3 KB)
│   │   ├── main.py   (6.3 KB)
│   │   ├── README.md   (0.6 KB)
│   │   ├── requirements.in   (0.0 KB)
│   │   └── requirements.txt   (0.8 KB)
│   ├── ur5/
│   │   └── README.md   (5.8 KB)
│   ├── inference.ipynb   (5.4 KB)
│   └── policy_records.ipynb   (3.4 KB)
├── packages/
│   └── openpi-client/
│       ├── src/
│       │   └── openpi_client/
│       │       ├── runtime/
│       │       │   ├── agents/
│       │       │   │   └── policy_agent.py   (0.5 KB)
│       │       │   ├── agent.py   (0.5 KB)
│       │       │   ├── environment.py   (1.1 KB)
│       │       │   ├── runtime.py   (3.1 KB)
│       │       │   └── subscriber.py   (0.5 KB)
│       │       ├── __init__.py   (0.0 KB)
│       │       ├── action_chunk_broker.py   (1.4 KB)
│       │       ├── base_policy.py   (0.3 KB)
│       │       ├── image_tools.py   (2.4 KB)
│       │       ├── image_tools_test.py   (1.4 KB)
│       │       ├── msgpack_numpy.py   (2.0 KB)
│       │       ├── msgpack_numpy_test.py   (1.6 KB)
│       │       └── websocket_client_policy.py   (2.1 KB)
│       └── pyproject.toml   (0.4 KB)
├── scripts/
│   ├── docker/
│   │   ├── compose.yml   (0.8 KB)
│   │   ├── install_docker_ubuntu22.sh   (1.5 KB)
│   │   ├── install_nvidia_container_toolkit.sh   (1.0 KB)
│   │   └── serve_policy.Dockerfile   (1.9 KB)
│   ├── __init__.py   (0.0 KB)
│   ├── compute_norm_stats.py   (4.5 KB)
│   ├── eval_on_libero.sh   (0.3 KB)
│   ├── eval_on_libero_plus.py   (11.1 KB)
│   ├── openloop.py   (3.6 KB)
│   ├── serve_policy.py   (4.5 KB)
│   ├── server.sh   (0.4 KB)
│   ├── train.py   (13.5 KB)
│   └── train.sh   (0.2 KB)
├── src/
│   └── openpi/
│       ├── models/
│       │   ├── __init__.py   (0.0 KB)
│       │   ├── acot_vla.py   (41.4 KB)
│       │   ├── gemma.py   (18.3 KB)
│       │   ├── gemma_fast.py   (15.7 KB)
│       │   ├── lora.py   (5.4 KB)
│       │   ├── lora_test.py   (3.2 KB)
│       │   ├── model.py   (12.3 KB)
│       │   ├── model_test.py   (3.0 KB)
│       │   ├── pi0.py   (16.9 KB)
│       │   ├── pi0_fast.py   (13.0 KB)
│       │   ├── pi0_test.py   (1.6 KB)
│       │   ├── siglip.py   (12.2 KB)
│       │   ├── tokenizer.py   (6.6 KB)
│       │   ├── tokenizer_test.py   (0.8 KB)
│       │   └── vit.py   (10.3 KB)
│       ├── policies/
│       │   ├── agilex_fk.py   (9.5 KB)
│       │   ├── agilex_policy.py   (9.0 KB)
│       │   ├── aloha_policy.py   (11.3 KB)
│       │   ├── arx_policy.py   (7.7 KB)
│       │   ├── droid_policy.py   (3.2 KB)
│       │   ├── go1_policy.py   (7.5 KB)
│       │   ├── go2_policy.py   (9.9 KB)
│       │   ├── libero_policy.py   (6.8 KB)
│       │   ├── policy.py   (4.4 KB)
│       │   ├── policy_config.py   (2.9 KB)
│       │   ├── policy_test.py   (1.1 KB)
│       │   └── vlabench_policy.py   (5.9 KB)
│       ├── serving/
│       │   └── websocket_policy_server.py   (3.1 KB)
│       ├── shared/
│       │   ├── __init__.py   (0.0 KB)
│       │   ├── array_typing.py   (3.3 KB)
│       │   ├── download.py   (8.0 KB)
│       │   ├── download_test.py   (1.5 KB)
│       │   ├── image_tools.py   (1.9 KB)
│       │   ├── image_tools_test.py   (1.4 KB)
│       │   ├── nnx_utils.py   (2.9 KB)
│       │   ├── normalize.py   (5.5 KB)
│       │   └── normalize_test.py   (0.8 KB)
│       ├── training/
│       │   ├── checkpoints.py   (6.3 KB)
│       │   ├── config.py   (92.2 KB)
│       │   ├── data_loader.py   (22.5 KB)
│       │   ├── data_loader_test.py   (2.5 KB)
│       │   ├── droid_rlds_dataset.py   (7.0 KB)
│       │   ├── optimizer.py   (3.1 KB)
│       │   ├── sampler.py   (4.0 KB)
│       │   ├── sharding.py   (4.1 KB)
│       │   ├── utils.py   (1.5 KB)
│       │   └── weight_loaders.py   (6.5 KB)
│       ├── __init__.py   (0.0 KB)
│       ├── conftest.py   (0.3 KB)
│       ├── py.typed   (0.0 KB)
│       ├── transforms.py   (19.3 KB)
│       └── transforms_test.py   (4.1 KB)
├── .dockerignore   (0.0 KB)
├── .gitignore   (3.3 KB)
├── .gitmodules   (0.2 KB)
├── .pre-commit-config.yaml   (0.3 KB)
├── .python-version   (0.0 KB)
├── LICENSE   (11.3 KB)
├── pyproject.toml   (3.0 KB)
├── README.md   (8.0 KB)
└── uv.lock   (847.8 KB)
```
