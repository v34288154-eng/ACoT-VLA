# 模块地图（自动生成：类/函数 → 文件:行号）

> 生成时间：2026-09-05 17:16；基线 commit：`c323377`。
> 覆盖 ('src/openpi', 'packages', 'scripts', 'examples') 下的公开符号：65 个模块、184 个类、115 个函数/方法。
> 由 `python tools/module_map.py --repo <repo> --commit <sha>` 生成，请勿手改。

### src/openpi/models/acot_vla.py
- `MLP` (行 22)
- `LearnableQueryExtractor` (行 35)
- `AttentionPoolingExtractor` (行 105)
- `DownsampleExtractor` (行 156)
- `UnifiedAttentionModule` (行 242)
- `ACOTConfig` (行 267)
    - `model_type` (行 297)
    - `create` (行 304)
    - `inputs_spec` (行 308)
    - `get_freeze_filter` (行 333)
- `ACOT_VLA` (行 376)
    - `embed_prefix` (行 516)
    - `embed_suffix` (行 551)
    - `compute_loss` (行 695)
    - `sample_actions` (行 795)

### src/openpi/models/gemma.py
- `Config` (行 45)
- `get_config` (行 58) — Returns config for specified gemma variant.
- `RMSNorm` (行 160)
- `Embedder` (行 182) — Embedder module.
    - `setup` (行 188)
    - `encode` (行 195)
    - `decode` (行 200)
- `Attention` (行 205) — Attention module.
- `FeedForward` (行 300) — Feed forward module.
- `Block` (行 331) — Transformer block.
- `Module` (行 387) — Transformer model, supporting a mixture of different weights for different tokens.
    - `setup` (行 397)
    - `embed` (行 432)
    - `init` (行 460)

### src/openpi/models/gemma_fast.py
- `get_config` (行 35) — Returns config for specified gemma variant.
- `Einsum` (行 77)
- `RMSNorm` (行 88)
- `Embedder` (行 102) — Embedder module.
    - `setup` (行 108)
    - `encode` (行 115)
    - `decode` (行 120)
- `Attention` (行 125) — Attention module.
    - `setup` (行 137)
- `Block` (行 228) — Transformer block.
    - `setup` (行 242)
- `Module` (行 279) — gemma model.
    - `init` (行 420)

### src/openpi/models/lora.py
- `LoRAConfig` (行 12) — Configuration for LoRA.
    - `scaling_value` (行 29)
- `Einsum` (行 33) — Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum.
    - `setup` (行 43)
- `FeedForward` (行 88) — Feed forward module.
    - `setup` (行 96)

### src/openpi/models/model.py
- `convert_str_keys_to_int` (行 25)
- `ModelType` (行 41) — Supported model types.
- `Observation` (行 95) — Holds observations, i.e., inputs to the model.
    - `from_dict` (行 122)
    - `to_dict` (行 141)
- `preprocess_observation` (行 154) — Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
- `BaseModelConfig` (行 222) — Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    - `model_type` (行 236)
    - `create` (行 240)
    - `load` (行 243)
    - `inputs_spec` (行 255)
    - `fake_obs` (行 258)
    - `fake_act` (行 262)
- `BaseModel` (行 268) — Base class for all model implementations. Specific models should inherit from this class. They should call
    - `compute_loss` (行 278)
    - `sample_actions` (行 288)
- `restore_params` (行 291) — Restores unstructured params PyTree from a checkpoint.

### src/openpi/models/pi0.py
- `make_attn_mask` (行 20) — Adapted from big_vision.
- `posemb_sincos` (行 49) — Computes sine-cosine positional embedding vectors for scalar positions.
- `Pi0Config` (行 68)
    - `model_type` (行 92)
    - `create` (行 98)
    - `inputs_spec` (行 102)
    - `get_freeze_filter` (行 126)
- `Pi0` (行 169)
    - `embed_prefix` (行 209)
    - `embed_suffix` (行 243)
    - `compute_loss` (行 292)
    - `sample_actions` (行 320)

### src/openpi/models/pi0_fast.py
- `make_attn_mask` (行 22) — Adapted from big_vision.
- `left_to_right_align` (行 51) — Converts input from left-align to right-aligned.
- `put_along_last_axis` (行 66) — Like np.put_along_axis(..., axis=-1), since jax is missing it.
- `Pi0FASTConfig` (行 76)
    - `model_type` (行 87)
    - `create` (行 91)
    - `inputs_spec` (行 95)
    - `get_freeze_filter` (行 121)
- `Pi0FAST` (行 128)
    - `embed_inputs` (行 154)
    - `compute_loss` (行 192)
    - `sample_actions` (行 230)

### src/openpi/models/siglip.py
- `posemb_sincos_2d` (行 27) — Follows the MoCo v3 logic.
- `get_posemb` (行 40)
- `MlpBlock` (行 53) — Transformer MLP / feed-forward block.
- `Encoder1DBlock` (行 75) — Single transformer encoder block (MHSA + MLP).
- `Encoder` (行 111) — Transformer Model Encoder for sequence to sequence translation.
- `MAPHead` (行 164) — Multihead Attention Pooling.
- `Module` (行 293) — Factory function, because linen really don't like what I'm doing!
- `decode_variant` (行 298) — Converts a string like "B" or "B/32" into a params dict.

### src/openpi/models/tokenizer.py
- `PaligemmaTokenizer` (行 10)
    - `tokenize` (行 18)
- `FASTTokenizer` (行 47)
    - `tokenize` (行 60)
    - `extract_actions` (行 115)

### src/openpi/models/vit.py
- `IdentityLayer` (行 31) — Identity layer, convenient for giving a name to an array.
- `AddPositionEmbs` (行 39) — Adds learned positional embeddings to the inputs.
- `MlpBlock` (行 66) — Transformer MLP / feed-forward block.
- `Encoder1DBlock` (行 104) — Transformer encoder layer.
- `Encoder` (行 160) — Transformer Model Encoder for sequence to sequence translation.
- `VisionTransformer` (行 219) — VisionTransformer.

### src/openpi/policies/agilex_fk.py
- `C_PiperForwardKinematics` (行 7)
    - `CalFK` (行 111)
- `qpos_to_eef_pos` (行 144) — Convert 14-dimensional qpos to end-effector positions (6 DOF + 1 gripper for each arm).
- `batch_qpos_to_eef_pos` (行 200) — Convert a batch of 14-dimensional qpos to end-effector positions.

### src/openpi/policies/agilex_policy.py
- `AgilexInputs` (行 16) — Inputs for the Agilex policy.
- `AgilexOutputs` (行 120) — Outputs for the Agilex policy.
- `AgilexACOTInputs` (行 129) — Inputs for the Agilex policy.
- `AgilexACOTOutputs` (行 232) — Outputs for the Agilex policy.

### src/openpi/policies/aloha_policy.py
- `make_aloha_example` (行 10) — Creates a random input example for the Aloha policy.
- `AlohaInputs` (行 25) — Inputs for the Aloha policy.
- `AlohaACOTInputs` (行 90) — Inputs for the Aloha policy.
- `AlohaOutputs` (行 169) — Outputs for the Aloha policy.
- `AlohaACOTOutputs` (行 182) — Outputs for the Aloha policy.

### src/openpi/policies/arx_policy.py
- `ARXInputs` (行 15) — Inputs for the Go1 policy.
- `ARXOutputs` (行 96) — Outputs for the Go1 policy.
- `ARXACOTInputs` (行 104) — Inputs for the Go1 policy.
- `ARXACOTOutputs` (行 201) — Outputs for the Go1 policy.

### src/openpi/policies/droid_policy.py
- `make_droid_example` (行 10) — Creates a random input example for the Droid policy.
- `DroidInputs` (行 31)
- `DroidOutputs` (行 78)

### src/openpi/policies/go1_policy.py
- `Go1Inputs` (行 15) — Inputs for the Go1 policy.
- `Go1Outputs` (行 96) — Outputs for the Go1 policy.
- `Go1ACOTInputs` (行 104) — Inputs for the Go1 policy.
- `Go1ACOTOutputs` (行 199) — Outputs for the Go1 policy.

### src/openpi/policies/go2_policy.py
- `Go2Inputs` (行 15) — Inputs for the Go2 policy.
- `Go2Outputs` (行 95) — Outputs for the Go2 policy.
- `Go2ACOTInputs` (行 103) — Inputs for the Go2 policy.
    - `slice_state_and_action` (行 122)
    - `random_inject_prompt` (行 138)
- `Go2ACOTOutputs` (行 238) — Outputs for the Go2 policy.

### src/openpi/policies/libero_policy.py
- `make_libero_example` (行 10) — Creates a random input example for the Libero policy.
- `LiberoInputs` (行 30) — This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
- `LiberoOutputs` (行 87) — This class is used to convert outputs from the model back the the dataset specific format. It is
- `LiberoACOTInputs` (行 103)
- `LiberoACOTOutputs` (行 155)

### src/openpi/policies/policy.py
- `Policy` (行 23)
    - `infer` (行 42)
    - `post_process` (行 72)
    - `metadata` (行 93)
- `PolicyRecorder` (行 97) — Records the policy's behavior to disk.
    - `infer` (行 109)

### src/openpi/policies/policy_config.py
- `create_trained_policy` (行 15) — Create a policy from a trained checkpoint.

### src/openpi/policies/vlabench_policy.py
- `quat2euler` (行 13)
- `make_libero_example` (行 19) — Creates a random input example for the Libero policy.
- `VLABenchInputs` (行 39)
- `VLABenchOutputs` (行 87)
- `VLABenchACOTInputs` (行 94)
- `VLABenchACOTOutputs` (行 153)

### src/openpi/serving/websocket_policy_server.py
- `WebsocketPolicyServer` (行 15) — Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.
    - `serve_forever` (行 34)
    - `run` (行 37)

### src/openpi/shared/array_typing.py
- `typecheck` (行 50)
- `disable_typechecking` (行 55)
- `check_pytree_equality` (行 62) — Checks that two PyTrees have the same structure and optionally checks shapes and dtypes. Creates a much nicer

### src/openpi/shared/download.py
- `get_cache_dir` (行 24)
- `maybe_download` (行 35) — Download a file or directory from a remote filesystem to the local cache, and return the local path.

### src/openpi/shared/image_tools.py
- `resize_with_pad` (行 11) — Replicates tf.image.resize_with_pad. Resizes an image to a target height and width without distortion

### src/openpi/shared/nnx_utils.py
- `module_jit` (行 19) — A higher-order function to JIT-compile `nnx.Module` methods, freezing the module's state in the process.
- `PathRegex` (行 51) — NNX Filter that matches paths using a regex.
- `state_map` (行 70) — Apply a function to the leaves of the state that match the filter.

### src/openpi/shared/normalize.py
- `NormStats` (行 10)
- `RunningStats` (行 17) — Compute running statistics of a batch of vectors.
    - `update` (行 30)
    - `get_statistics` (行 74)
- `serialize_json` (行 125) — Serialize the running statistics to a JSON string.
- `deserialize_json` (行 130) — Deserialize the running statistics from a JSON string.
- `save` (行 135) — Save the normalization stats to a directory.
- `load` (行 142) — Load the normalization stats from a directory.

### src/openpi/training/checkpoints.py
- `initialize_checkpoint_dir` (行 20)
- `save_state` (行 65)
- `restore_state` (行 95)
- `load_norm_stats` (行 116)
- `Callback` (行 126)
- `CallbackHandler` (行 130) — A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring.
    - `save` (行 133)
    - `async_save` (行 137)
    - `restore` (行 140)
- `CallbackSave` (行 146)
- `CallbackRestore` (行 151)

### src/openpi/training/config.py
- `AssetsConfig` (行 43) — Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.
- `DataConfig` (行 70)
- `GroupFactory` (行 108)
- `ModelTransformFactory` (行 114) — Creates model transforms for standard pi0 models.
- `DataConfigFactory` (行 191)
    - `create` (行 200)
    - `create_base_config` (行 203)
- `FakeDataConfig` (行 254)
    - `create` (行 258)
- `SimpleDataConfig` (行 263)
    - `create` (行 270)
- `LeRobotAlohaDataConfig` (行 279)
    - `create` (行 308)
- `LeRobotLiberoDataConfig` (行 332) — This config is used to configure transforms that are applied at various parts of the data pipeline.
    - `create` (行 342)
- `LeRobotVLABenchDataConfig` (行 408)
    - `create` (行 411)
- `LeRobotACOTVLABenchDataConfig` (行 451)
    - `create` (行 457)
- `LerobotACOTGo1DataConfig` (行 504) — Configuration for the Go1 robot dataset.
    - `create` (行 550)
- `LerobotACOTGo2DataConfig` (行 583) — Configuration for the Go2 robot dataset.
    - `create` (行 633)
- `LeRobotACOTLiberoDataConfig` (行 666)
    - `create` (行 671)
- `LeRobotACOTLiberoPlusDataConfig` (行 713)
    - `create` (行 719)
- `RLDSDroidDataConfig` (行 762) — Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    - `create` (行 771)
- `LeRobotDROIDDataConfig` (行 814) — Example data config for custom DROID dataset in LeRobot format.
    - `create` (行 821)
- `LerobotAgilexDataConfig` (行 852) — Configuration for the Agilex robot dataset.
    - `create` (行 893)
- `LerobotACOTAgilexDataConfig` (行 928) — Configuration for the Agilex robot dataset.
    - `create` (行 967)
- `LerobotARXDataConfig` (行 1001) — Configuration for the Lerobot ARX dataset.
    - `create` (行 1039)
- `LerobotACOTARXDataConfig` (行 1074) — Configuration for the Lerobot ARX dataset.
    - `create` (行 1113)
- `TrainConfig` (行 1150)
    - `assets_dirs` (行 1216)
    - `checkpoint_dir` (行 1221)
    - `trainable_filter` (行 1231)
- `cli` (行 1947)
- `get_config` (行 1951) — Get a config by name.

### src/openpi/training/data_loader.py
- `Dataset` (行 21) — Interface for a dataset with random access.
- `SafeDataset` (行 31)
- `IterableDataset` (行 52) — Interface for an iterable dataset.
- `DataLoader` (行 62) — Interface for a data loader.
    - `data_config` (行 65)
- `TransformedDataset` (行 73)
- `IterableTransformedDataset` (行 101)
- `FakeDataset` (行 135)
- `create_torch_dataset` (行 166) — Create a dataset for training.
- `create_rlds_dataset` (行 229)
- `transform_dataset` (行 246) — Transform the dataset by applying the data transforms.
- `transform_iterable_dataset` (行 268) — Transform the dataset by applying the data transforms.
- `create_data_loader` (行 297) — Create a data loader for training.
- `create_torch_data_loader` (行 332) — Create a data loader for training.
- `create_rlds_data_loader` (行 389) — Create an RLDS data loader for training.
- `TorchDataLoader` (行 427)
    - `torch_loader` (行 492)
- `RLDSDataLoader` (行 543) — Shallow wrapper around the DROID data loader to make it compatible with openpi.
- `DataLoaderImpl` (行 587)
    - `data_config` (行 592)
- `DataLoaderACOTImpl` (行 599)
    - `data_config` (行 604)

### src/openpi/training/droid_rlds_dataset.py
- `DroidActionSpace` (行 12) — Action space for DROID dataset.
- `DroidRldsDataset` (行 19)

### src/openpi/training/optimizer.py
- `LRScheduleConfig` (行 11)
    - `create` (行 12)
- `CosineDecaySchedule` (行 16) — Cosine decay schedule with warmup.
    - `create` (行 24)
- `RsqrtDecaySchedule` (行 35) — Inverse square root decay schedule with warmup.
    - `create` (行 42)
- `OptimizerConfig` (行 57)
    - `create` (行 58)
- `AdamW` (行 66) — AdamW optimizer.
    - `create` (行 75)
- `SGD` (行 88) — SGD optimizer.
    - `create` (行 95)
- `create_optimizer` (行 104)

### src/openpi/training/sampler.py
- `get_base_dataset` (行 7)
- `sample_subtask` (行 12)
- `FrameSampler` (行 66) — Custom sampler that only samples data indices falling within specified intervals
    - `parse_dataset` (行 74)
    - `sample_frames` (行 84)

### src/openpi/training/sharding.py
- `make_mesh` (行 17)
- `set_mesh` (行 27) — Plumbing the mesh deep into the module tree is extremeley cumbersome; until the JAX team lands a better API, a
- `activation_sharding_constraint` (行 40)
- `fsdp_sharding` (行 48) — Apply FSDP sharding to a pytree of arrays based on the mesh shape.

### src/openpi/training/utils.py
- `TrainState` (行 15)
- `tree_to_info` (行 27) — Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
- `array_tree_to_info` (行 36) — Converts a PyTree of arrays into a human-readable string for logging.
- `count_parameters` (行 40) — Count total number of parameters in a JAX PyTree.

### src/openpi/training/weight_loaders.py
- `WeightLoader` (行 17)
    - `load` (行 18)
- `NoOpWeightLoader` (行 32)
    - `load` (行 33)
- `CheckpointWeightLoader` (行 38) — Loads an entire set of weights from a checkpoint.
    - `load` (行 50)
- `ACOTCheckpointWeightLoader` (行 57)
    - `load` (行 60)
- `PaliGemmaWeightLoader` (行 84) — Loads weights from the official PaliGemma checkpoint.
    - `load` (行 91)

### src/openpi/transforms.py
- `DataTransformFn` (行 26)
- `Group` (行 42) — A group of transforms.
    - `push` (行 51)
- `CompositeTransform` (行 65) — A composite transform that applies a sequence of transforms in order.
- `compose` (行 76) — Compose a sequence of transforms into a single transform.
- `RepackTransform` (行 82) — Repacks an input dictionary into a new dictionary.
- `InjectDefaultPrompt` (行 107)
- `Normalize` (行 117)
- `Unnormalize` (行 151)
- `ResizeImages` (行 187)
    - `to_numpy` (行 192)
- `SubsampleActions` (行 206)
- `DeltaActions` (行 215) — Repacks absolute actions into delta action space.
- `AbsoluteActions` (行 237) — Repacks delta actions into absolute action space.
- `ACOTDeltaActions` (行 258) — Repacks absolute actions into delta action space.
- `ACOTAbsoluteActions` (行 278) — Repacks delta actions into absolute action space.
- `TokenizePrompt` (行 297)
- `TokenizeFASTInputs` (行 319)
- `ExtractFASTActions` (行 341)
- `PromptFromLeRobotTask` (行 359) — Extracts a prompt from the current LeRobot dataset task.
- `PromptFromHighlevelInstruction` (行 376) — Extracts a prompt from the current LeRobot dataset task.
- `PadStatesAndActions` (行 405) — Zero-pads states and actions to the model action dimension.
- `ACOTPadStatesAndActions` (行 417) — Zero-pads states and actions to the model action dimension.
- `flatten_dict` (行 430) — Flatten a nested dictionary. Uses '/' as the separator.
- `unflatten_dict` (行 435) — Unflatten a flattened dictionary. Assumes that '/' was used as a separator.
- `transform_dict` (行 440) — Transform the structure of a nested dictionary using a set of patterns.
- `apply_tree` (行 494)
- `pad_to_dim` (行 513) — Pad an array to the target dimension with zeros along the specified axis.
- `make_bool_mask` (行 523) — Make a boolean mask for the given dimensions.

### packages/openpi-client/src/openpi_client/action_chunk_broker.py
- `ActionChunkBroker` (行 10) — Wraps a policy to return action chunks one-at-a-time.
    - `infer` (行 27)
    - `reset` (行 47)

### packages/openpi-client/src/openpi_client/base_policy.py
- `BasePolicy` (行 5)
    - `infer` (行 7)
    - `reset` (行 10)

### packages/openpi-client/src/openpi_client/image_tools.py
- `convert_to_uint8` (行 5) — Converts an image to uint8 if it is a float image.
- `resize_with_pad` (行 15) — Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

### packages/openpi-client/src/openpi_client/msgpack_numpy.py
- `pack_array` (行 21)
- `unpack_array` (行 43)

### packages/openpi-client/src/openpi_client/runtime/agent.py
- `Agent` (行 4) — An Agent is the thing with agency, i.e. the entity that makes decisions.
    - `get_action` (行 12)
    - `reset` (行 16)

### packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py
- `PolicyAgent` (行 7) — An agent that uses a policy to determine actions.
    - `get_action` (行 14)
    - `reset` (行 17)

### packages/openpi-client/src/openpi_client/runtime/environment.py
- `Environment` (行 4) — An Environment represents the robot and the environment it inhabits.
    - `reset` (行 12)
    - `is_episode_complete` (行 19)
    - `get_observation` (行 27)
    - `apply_action` (行 31)

### packages/openpi-client/src/openpi_client/runtime/runtime.py
- `Runtime` (行 10) — The core module orchestrating interactions between key components of the system.
    - `run` (行 32)
    - `run_in_new_thread` (行 40)
    - `mark_episode_complete` (行 46)

### packages/openpi-client/src/openpi_client/runtime/subscriber.py
- `Subscriber` (行 4) — Subscribes to events in the runtime.
    - `on_episode_start` (行 11)
    - `on_step` (行 15)
    - `on_episode_end` (行 19)

### packages/openpi-client/src/openpi_client/websocket_client_policy.py
- `WebsocketClientPolicy` (行 12) — Implements the Policy interface by communicating with a server over websocket.
    - `get_server_metadata` (行 26)
    - `infer` (行 44)
    - `reset` (行 54)

### scripts/compute_norm_stats.py
- `RemoveStrings` (行 19)
- `create_torch_dataloader` (行 24)
- `create_rlds_dataloader` (行 61)
- `main` (行 89)

### scripts/eval_on_libero_plus.py
- `Args` (行 22)
- `eval_libero` (行 50)

### scripts/serve_policy.py
- `EnvMode` (行 14) — Supported environments.
- `Checkpoint` (行 27) — Load a policy from a trained checkpoint.
- `Default` (行 37) — Use the default policy for the given environment.
- `Args` (行 42) — Arguments for the serve_policy script.
- `create_default_policy` (行 94) — Create a default policy for the given environment.
- `create_policy` (行 103) — Create a policy from the given arguments.
- `main` (行 114)

### scripts/train.py
- `init_logging` (行 31) — Custom logging format for better readability.
- `init_wandb` (行 50)
- `init_train_state` (行 85)
- `train_step` (行 137)
- `acot_train_step` (行 194)
- `main` (行 250)

### examples/aloha_real/convert_aloha_data_to_lerobot.py
- `DatasetConfig` (行 23)
- `create_empty_dataset` (行 34)
- `get_cameras` (行 128)
- `has_velocity` (行 134)
- `has_effort` (行 139)
- `load_raw_images_per_camera` (行 144)
- `load_raw_episode_data` (行 165)
- `populate_dataset` (行 193)
- `port_aloha` (行 229)

### examples/aloha_real/env.py
- `AlohaRealEnvironment` (行 11) — An environment for an Aloha robot on real hardware.
    - `reset` (行 27)
    - `is_episode_complete` (行 31)
    - `get_observation` (行 35)
    - `apply_action` (行 56)

### examples/aloha_real/main.py
- `Args` (行 14)
- `main` (行 24)

### examples/aloha_real/real_env.py
- `RealEnv` (行 18) — Environment for real robot bi-manual manipulation
    - `setup_robots` (行 62)
    - `get_qpos` (行 66)
    - `get_qvel` (行 79)
    - `get_effort` (行 88)
    - `get_images` (行 95)
    - `set_gripper_pose` (行 98)
    - `get_observation` (行 128)
    - `get_reward` (行 136)
    - `reset` (行 139)
    - `step` (行 150)
- `get_action` (行 163)
- `make_real_env` (行 175)

### examples/aloha_real/robot_utils.py
- `ImageRecorder` (行 19)
    - `image_cb` (行 48)
    - `image_cb_cam_high` (行 72)
    - `image_cb_cam_low` (行 76)
    - `image_cb_cam_left_wrist` (行 80)
    - `image_cb_cam_right_wrist` (行 84)
    - `get_images` (行 88)
    - `print_diagnostics` (行 100)
- `Recorder` (行 112)
    - `puppet_state_cb` (行 141)
    - `puppet_arm_commands_cb` (行 149)
    - `puppet_gripper_commands_cb` (行 154)
    - `print_diagnostics` (行 159)
- `get_arm_joint_positions` (行 172)
- `get_arm_gripper_positions` (行 176)
- `move_arms` (行 180)
- `move_grippers` (行 193)
- `setup_puppet_bot` (行 214)
- `setup_master_bot` (行 221)
- `set_standard_pid_gains` (行 227)
- `set_low_pid_gains` (行 232)
- `torque_off` (行 237)
- `torque_on` (行 242)
- `sync_puppet_to_master` (行 248)

### examples/aloha_real/video_display.py
- `VideoDisplay` (行 7) — Displays video frames.
    - `on_episode_start` (行 15)
    - `on_step` (行 21)
    - `on_episode_end` (行 34)

### examples/aloha_sim/env.py
- `AlohaSimEnvironment` (行 9) — An environment for an Aloha robot in simulation.
    - `reset` (行 23)
    - `is_episode_complete` (行 30)
    - `get_observation` (行 34)
    - `apply_action` (行 41)

### examples/aloha_sim/main.py
- `Args` (行 15)
- `main` (行 29)

### examples/aloha_sim/saver.py
- `VideoSaver` (行 10) — Saves episode data.
    - `on_episode_start` (行 20)
    - `on_step` (行 24)
    - `on_episode_end` (行 30)

### examples/droid/convert_droid_data_to_lerobot.py
- `resize_image` (行 32)
- `main` (行 37)
- `get_camera_type` (行 180)
- `MP4Reader` (行 187)
    - `set_reading_parameters` (行 198)
    - `get_frame_resolution` (行 214)
    - `get_frame_count` (行 219)
    - `set_frame_index` (行 224)
    - `read_camera` (行 241)
    - `disable_camera` (行 269)
- `RecordedMultiCameraWrapper` (行 274)
    - `read_cameras` (行 296)
- `get_hdf5_length` (行 329)
- `load_hdf5_to_dict` (行 351)
- `TrajectoryReader` (行 369)
    - `length` (行 378)
    - `read_timestep` (行 381)
    - `close` (行 400)
- `load_trajectory` (行 404)

### examples/droid/main.py
- `Args` (行 27)
- `prevent_keyboard_interrupt` (行 55) — Temporarily prevent keyboard interrupts by delaying them until after the protected code.
- `main` (行 73)

### examples/libero/convert_libero_data_to_lerobot.py
- `main` (行 37)

### examples/libero/main.py
- `Args` (行 27)
- `eval_libero` (行 54)

### examples/simple_client/main.py
- `EnvMode` (行 17) — Supported environments.
- `Args` (行 27) — Command line arguments.
- `TimingRecorder` (行 44) — Records timing measurements for different keys.
    - `record` (行 50)
    - `get_stats` (行 56)
    - `print_all_stats` (行 70)
    - `write_parquet` (行 109)
- `main` (行 117)
