# Sharpa 番茄采摘数据接入 T-Rex

结论：现有数据具备三路 RGB、双臂/双手状态与目标、十指力/力矩和十指形变图，可以作为 T-Rex post-training 的输入。它不能直接交给官方 LeRobot v3 loader：版本、字段和动作表示均不同。本适配器直接读取原来的 v2.1 Parquet/MP4，复用官方 `TRexLeRobotDataset.collate_fn`，不复制 RGB，不转存十路触觉视频。

**已从提供的数采工程找到 FR3 名义运动学模型并补齐配置。** `robot/dual_fr3_kinematics.urdf` 直接来自工程的 Pico 资源目录。数据适配采用每臂 `fr3v2_link0 → fr3v2_link8`，即法兰参考帧；关节别名映射由采集工程 `teleop_core.contract` 确认。机械臂界面以 link8 为工具参考，Pico IK 则使用 link7，本适配明确采用前者。GELLO 采集的是关节命令，因此这里是新策略的笛卡尔动作约定，不声称原始 bag 已记录了法兰目标位姿。独立 FK、坐标还原和真实关节数据检查通过；这不是与实机笛卡尔测量对比的几何标定。证据见 `机器人模型与末端说明.md`、`collection_reference.json` 和 `fr3_validation.json`。

## 已检查的数据

| 数据目录 | 轨迹 | 帧数 |
|---|---:|---:|
| A/121_430_data_lerobot | 168 | 157,442 |
| A/54_120_data_lerobot | 41 | 37,145 |
| B/130_423_data_lerobot | 34 | 40,221 |
| B/56_106_data_lerobot | 8 | 10,607 |
| 合计 | 251 | 245,415 |

30 Hz，约 2.27 小时。原始采集 ID 共 251 个，元数据中未发现重复 ID。目录名中的编号范围不能当作实际轨迹数。

全量读过 245,415 帧的状态、动作和十指六轴力，未发现非有限数值、帧数不符或视频文件缺失。采集侧 251 份 `collection_state.json` 也已检查：A 的 209 条标注为“优等”，B 的 42 条标注为“一般”，均属于相同工作站。B 不应直接视为失败数据。可以先用 A 做第一轮实验，再比较加入 B 的效果；`config.a_only.json` 提供了仅 A 的输入根目录，`config.example.json` 包含 A+B。

## 对应关系

| 原数据 | T-Rex 输入 | 处理 |
|---|---|---|
| `observation.state[116]` | `observation.state[62]` | 按关节名字选取位置，去除速度，双臂 FK，手指保留位置 |
| `action[58]` | `action[16,62]` | 每臂 7 关节目标 → 末端目标；生成 16 步 delta-base 块 |
| `observation.images.cam0` | `observation.images.head` | 高位相机 |
| `observation.images.cam1` | `observation.images.wrist_right` | 右腕相机 |
| `observation.images.cam2` | `observation.images.wrist_left` | 左腕相机 |
| 十个 `*.wrench[6]` | `observation.tactile_f6[16,10,6]` | 左手后右手；每手 thumb/index/middle/ring/pinky；Fx/Fy/Fz/Tx/Ty/Tz |
| 十个 `*.deformation[240,240]` | 十个 `observation.tactile_deform.*` | 按 Parquet 行组读取 uint8，除以 255，交给官方 collate |
| `audit.original_timestamp_ns` | 连续片段边界 | 动作块、触觉历史和 FLARE 未来图均不跨剪切点 |

相机角色有采集记录证据：251 份原始记录 `collection_state.json/device_identities` 均写明 cam0=cam_high、cam1=cam_right_wrist、cam2=cam_left_wrist；见 `source_provenance.json`、四个代表样本的 `camera_provenance.json` 和相机预览。没有把左右腕按编号猜测。

62 维排列严格为：左臂 `[xyz, rotation_col1, rotation_col2]` 9 维 + 左手 22 维 + 右臂 9 维 + 右手 22 维。每个块以**块起始帧的实测末端位姿**为基准：`dp=R0.T@(p_target-p0)`，`dR=R0.T@R_target`；手指目标保持绝对关节角。不能简单把 58 维补成 62 维，也不能改成相邻帧差分。

`source_timestamp_ns` 是整理后的时间轴。连续性检查使用保留的原始时间，默认 >1.5/30 秒或时间倒退即分段；片段开头复制首帧填触觉历史，片段尾复制最后目标填动作块。FLARE 越界帧使用官方 collate 的回退规则。实测 A/121 子集首条轨迹有 8 段；忽略剪切点会给模型制造错误的时序监督。

## 文件和接入步骤

- `sharpa_core.py`：按名字选维、展开 URDF 正运动学、62 维动作、片段边界。
- `prepare_sharpa.py`：生成较小的数值缓存、索引和训练集归一化统计。
- `sharpa_dataset.py`：按需读视频/形变图；复用官方 collate，不改模型和损失。
- `install_adapter.py`：给官方 `train.py` 新增 `--data_format sharpa`，保留原来的 JSON/LeRobot 入口，备份修改前的文件。
- `train_sharpa.sh`：沿用官方 post-training 主要参数；机器环境参数由训练端填写。
- `validate_adapter.py`：数值回归、准备流程测试、真实数据读取测试。
- `audit_sharpa.py`：全量检查数值列与视频文件完整性，抽查每个子集首尾轨迹的形变数组。
- `validate_fr3_model.py`：工程 FR3 模型、真实关节转换和已准备缓存的数值核对；独立数学核对使用 SciPy。
- `validate_prepared.py`：读取准备好的训练/验证样本，运行官方 collate；图像 token 使用测试替身。
- `relocate_prepared.py`：将准备好的缓存重连到训练机器的数据挂载目录。

在官方 T-Rex 环境中运行；依赖沿用官方 `pyproject.toml`（Python 3.10、PyTorch 2.6.0、NumPy、PyArrow、OpenCV、Pillow 等）。本适配器不需要安装 LeRobot，训练仍需要官方项目其余依赖。

```bash
python /path/to/trex_sharpa_adapter/install_adapter.py --repo /path/to/T-Rex
```

复制 `config.example.json` 后填写：

1. `dataset_roots`：四个真实数据根目录；Linux 上应改成相应挂载路径。
2. 两臂的 `urdf/base_link/tip_link/joint_names` 已填好。URDF 路径相对配置文件解析，随代码包提供，不依赖 ROS/xacro。URDF 名字带 `fr3v2`，数据列名字带 `fr3`，二者按 joint1…joint7 对应。不要再次应用 GELLO 的方向校正、灵敏度系数；记录的是经过安全网关验证的 FR3 目标。
3. 部署时的动作还原和 IK 必须使用相同的 link8 法兰参考。不能直接沿用 Pico 默认的 link7 IK；两个点相差局部 z 方向 0.107 米。采用法兰坐标系时，无需推测 Sharpa 手掌或指尖 TCP 偏移。
4. Sharpa 驱动 schema 的 22 个关节名字和顺序与 loader 一致，单位为弧度，`hand_scale=1`。驱动将 SDK `content.F6` 原样发布为六轴 wrench，与官方同系列传感器路径一样保留原值，`wrench_scale=[1,1,1,1,1,1]`。没有通过统计值反推物理单位，也没有额外变换左右手触觉坐标轴。

```bash
python /path/to/trex_sharpa_adapter/prepare_sharpa.py \
  --config /path/to/config.json --output /path/to/prepared_sharpa
```

输出目录必须是新目录，防止混入旧缓存。中途失败的目录应由使用者检查后清理或另选新目录。准备结果包括 `manifest.json`、`stats_data.json`、`numeric/*.npz`。训练时仍需原 Parquet 和 MP4 可访问。

准备结果也会保存 `adapter_metadata.json` 和 `robot/`，记录关节顺序、动作参考和 URDF 内容摘要。安装脚本让官方训练器把这两项复制进每个 checkpoint，以便部署时恢复同一动作含义。

如果使用本次生成的 `prepared_sharpa` 缓存，在 Linux 训练端复制它与适配代码包后，按实际挂载目录重连路径：

本次缓存包含全部 251 条、245,415 帧；按源轨迹划分为 238 条训练、13 条验证，分别为 232,570/12,845 帧。连续片段 863 个。数值缓存约 230 MB，分位数统计采样 20,000 条训练帧，seed=42。

```bash
python /path/to/trex_sharpa_adapter/relocate_prepared.py \
  --prepared /path/to/prepared_sharpa --data-root /path/to/data-lerobot/sharpa
```

脚本先检查数据/视频都存在，再更新索引，并保存旧索引备份。数值缓存无需重新计算。随后把 `PREPARED_ROOT` 指向该目录即可；`robot/` 和 `adapter_metadata.json` 也应随缓存一起复制。

验证集在准备阶段按原始 `source_recording_id` 划分，默认 5%、seed=42。归一化只使用训练数据；动作 q01/q99 保持 `[16,62]`，当前力/力矩为 60 维。默认从训练帧均匀抽样 20,000 帧估计分位数，并记录抽样信息；这不是全量精确分位数。需要精确统计时，将 `stats_max_frames` 设为不小于全部训练帧数，并预留相应内存。

训练使用官方发布的 [midtrain checkpoint](https://huggingface.co/miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6)，设置 `--resume_source midtrain`，以保留触觉专家。不要改成 pretrain，否则官方代码会重新初始化触觉专家。设置 `--action_dim 62 --action_chunk 16 --use_robot_state 0`；完整命令见 `train_sharpa.sh`。已嵌入 VQ-VAE 的 checkpoint 由官方脚本读取 `training_args.json` 自动配置，不需要离线烘焙 tactile codes。checkpoint 的 `action_dim/action_chunk` 必须匹配，不能依赖官方的 shape mismatch 跳过逻辑。

`train_sharpa.sh` 需要填写 `TREX_ROOT/PREPARED_ROOT/ORIGIN_MODEL_PATH/RESUME_CHECKPOINT/DEFORM_ENCODER_PATH/OUTPUT_DIR/NUM_PROCESSES/ACCELERATE_CONFIG/TRAIN_BSZ`；这些路径与进程数保留给训练端配置。可以追加参数覆盖实验设置。本次没有评估服务器硬件或启动训练。

## 验证范围

- 适配器动作计算与官方 `utils/lerobot_common.py` 的逐步实现对比通过，覆盖随机旋转、平移和尾部填充。
- 剪切边界填充、固定嵌套 Arrow 数组、URDF 变换顺序测试通过。
- 合成小数据的完整 prepare 流程通过，验证了按原始轨迹分组、只用训练集算统计，以及动作分位数形状。
- 四个真实子集均抽取首条轨迹，读取首尾 RGB、十指形变和六轴力，运行了官方 collate。报告分别见 `validation*.json` / `validation_*/validation.json`。
- collate 测试用小型 processor 替身隔离 VLM 依赖；没有运行真实 Qwen processor、模型前向/反向或 CKPT 加载。
- 实际 FR3 URDF 已从工程中补齐；新增 `validate_fr3_model.py` 在四个子集的真实关节数据上核对了转换与还原，并与独立 SciPy FK 对比。没有采集侧同步笛卡尔位姿，不能把模型计算一致性解释为实机绝对位姿精度验证。
- 已检查全部 251 份准备后的数值缓存：维度及有限性通过，训练/验证采集 ID 无交叉。`prepared_batch_validation.json` 记录了使用工程 FR3 模型缓存的官方 collate 检查；`checkpoint_metadata_validation.json` 验证了独立的 checkpoint 元数据保存钩子。
- 原数据已有的 `conversion/verification.json` 均显示通过，但其中 `lossless_sensor_rows_compared=0`、`independent_alignment=false`。本次没有把该记录当作逐像素与原始 ROS bag 独立核验的证据。
- 数据目录里的 `validated` 指令流与格式校验不等于任务成功标签。本次没有复核所有示范是否完成采摘；post-training 数据应保留你认可的示范。

复现测试：

```bash
python /path/to/trex_sharpa_adapter/validate_adapter.py \
  --upstream /path/to/T-Rex \
  --root /path/to/sharpa/A/121_430_data_lerobot \
  --output /path/to/validation_run
```

此命令内部的合成 URDF 仅用于读取/接口测试，不会产出可用于真实训练的已标定缓存。

参考：[论文](https://arxiv.org/html/2606.17055v1)、[官方仓库](https://github.com/ZhuoyangLiu2005/T-Rex)、[动作语义](https://github.com/ZhuoyangLiu2005/T-Rex/blob/main/utils/lerobot_common.py)、[官方 loader](https://github.com/ZhuoyangLiu2005/T-Rex/blob/main/qwen_vla/lerobot_dataset.py)。检查日期 2026-10-23；所参考代码的内容校验值见 `upstream_reference.json`。
