# Isaac Sim / HandsOn 交接

从普通仿真启动开始。这个目录保存番茄场景相关脚本、机器人模型构建器、果串导入资产，以及单独标识的历史研究。日常采摘脚本和数据采集入口见 [Newton 主仿真](../README.md)。

## 普通仿真启动

使用 Linux、支持 Isaac Sim 5.1 的 NVIDIA GPU、与 Isaac Sim 匹配的 Isaac Lab 和 HandsOn。历史本机环境为 Isaac Lab 2.3.2，RTX 5090。先按 NVIDIA 和 HandsOn 上游说明完成安装并接受适用的许可。获取 HandsOn 到一个单独的目录，基线为提交 `be2b59aad016869e8cede1f485d64e3c2d191685`。[HandsOn 上游](https://github.com/TsinghuaC3I/HandsOn/tree/be2b59aad016869e8cede1f485d64e3c2d191685)

从本目录执行下面的命令。`HANDSON_ROOT` 指向专用于仿真的 HandsOn 工作副本。`ISAAC_PYTHON` 指向已安装 Isaac Sim、Isaac Lab、NumPy、SciPy、Pillow 和 PyYAML 的 Python。两份 overlay 文件是原仿真实验的本地扩展。

```bash
export HANDSON_ROOT=/path/to/HandsOn
export ISAAC_PYTHON=/path/to/isaac-env/bin/python
install -m 644 handson-overlay/{sharpa_tactile,default_appearance}.py \
  "$HANDSON_ROOT/handson/handson/sim/isaaclab/"
install -m 644 handson-overlay/sharpa_lerobot_bridge.py "$HANDSON_ROOT/handson/handson/data/"

RUN_DIR="$PWD/outputs/preview_$(date +%Y%m%d_%H%M%S)"
"$ISAAC_PYTHON" scripts/build_dual_fr3_sharpa_mjcf.py \
  --source ../tomato-pick-sim --output "$RUN_DIR/robot/dual_fr3_sharpa.xml"

"$ISAAC_PYTHON" scripts/smoke_isaac_tomato.py \
  --headless --enable_cameras --steps 2 --render-every 1 \
  --mjcf "$RUN_DIR/robot/dual_fr3_sharpa.xml" --output "$RUN_DIR"
```

检查 `isaac_scene_import.json` 的 58 个转动关节，以及 `isaac_tomato_camera.png`、`isaac_tomato_progress.png` 和 `isaac_tomato_scene.usda`。默认 `configs/sharpa_sim_home.json` 是本次交接建立的名义仿真初始姿态。水果资产与主场景配置由当前目录和 Newton 目录提供。已用现有 Isaac 环境跑完 2 个 PhysX 步，输出上述文件。Kit 关闭时记录超时，场景和图像已经写出。完整采摘仍需单独验收。

## 策略闭环与传感器

- `scripts/smoke_isaac_tomato.py --help` 列出 T-Rex、pi05、相机、接触和形变选项。先记录只观察的短跑，再接入已验证的兼容服务，分别检查三路 RGB、关节、触觉有效区、动作坐标系、单位和刷新率。使用新的 `--output` 目录保存每次实验。
- `scripts/trex_isaac_observation_client.py` 与 `scripts/pi05_isaac_policy_client.py` 是两条仿真策略接口。模型服务、检查点和归一化统计单独准备。Newton OpenPI 的绝对 TCP 合同需按主交接文档核对。
- 仿真触觉和接触力只按对应模型的输出解释。PhysX 接触范数、离线番茄 FEM 反力、TacMap 代理深度以及真机 Wrench 分别记录。报表注明各自的来源、单位和采样方式。

### 接入模型服务

用普通仿真启动命令生成的 MJCF，准备新的 `--output` 目录。下面是两个独立试验，运行时只选择一组。`TREX_MODEL_PYTHON` 和 `TREX_CHECKPOINT` 分别指向 T-Rex 环境和检查点。历史 T-Rex 6250 服务采用 62 维动作、16 步动作块、384×288 输入，开启触觉向量与形变图。Isaac pi05 客户端使用匹配的 Sharpa 58 维策略服务。历史两类模型的权重和归一化文件由团队渠道提供。

```bash
"$ISAAC_PYTHON" scripts/smoke_isaac_tomato.py \
  --headless --enable_cameras --mjcf "$RUN_DIR/robot/dual_fr3_sharpa.xml" \
  --output "$RUN_DIR/trex" --episode-seconds 2 \
  --trex-host 127.0.0.1 --trex-port 5687 --trex-closed-loop \
  --trex-python "$TREX_MODEL_PYTHON" --trex-checkpoint "$TREX_CHECKPOINT" \
  --trex-task "Pick a tomato from the bunch and place it in the target box."

"$ISAAC_PYTHON" scripts/smoke_isaac_tomato.py \
  --headless --enable_cameras --mjcf "$RUN_DIR/robot/dual_fr3_sharpa.xml" \
  --output "$RUN_DIR/pi05" --episode-seconds 2 \
  --pi05-closed-loop --pi05-url ws://127.0.0.1:8000
```

这些命令用于检查模型接口。复现对照实验要登记真实或名义初始姿态、相机方案、动作单位、触觉设置、种子和检查点。历史[有无触觉配对汇总](historical-evaluation/ablation_10_corrected/summary.json)的 10 个配对种子中只有 2 对数值有效。可另查 [T-Rex 策略比较](historical-evaluation/trex_sweep_summary/summary.json)。有无触觉配对使用 `scripts/evaluate_sharpa_isaac_success.py`，还需要两种训练权重、各自的 `summary.json` 及已授权的数据集。此次未重跑模型服务或配对实验。

## Episode 54 与单果诊断

[episode54-real-input/](episode54-real-input/) 是 Isaac 对真机关节记录的**混合来源**复现入口。目录包含仿真侧的刚体、原生变形体结果、分析脚本和[实验报告](episode54-real-input/实验报告.md)。报告中提到的原始真机文件、混合视频、历史文件名和完整运行目录按历史记录保留，需在已有权限的实验归档找原件。

| 团队侧输入 | 复现时的设置 |
| --- | --- |
| LeRobot A/121_430_data_lerobot，Episode 54 | `EPISODE54_DATASET` 指向含同源 Parquet 和三路 RGB 视频的数据集 |
| 951×58、30 Hz 的关节位置回放文件 | `EPISODE54_REPLAY_H5`，可依照下面的命令从只读数据集生成 |
| 正式左右腕部相机参数 | `EPISODE54_WRIST_JSON` 指向正式 `wrist_camera_calibration_palm_left.json` |
| 带相机的正式机器人 MJCF 和 cam0 对齐参数 | `EPISODE54_ALIGNMENT` 包含 `camera_assembly_blue_home_palm_left/dual_fr3_sharpa_camera_visuals.xml`、其网格及 `side_cam0_episode54_alignment.json` |
| 原回放的 home 配置 | HandsOn 副本内 `configs/sharpa_powder_weighing_real_home.json` |
| 单果场景的原始关节位置 CSV | `EPISODE54_REAL_JOINT_CSV`，只在检查姿态和几何探测时使用 |

```bash
export EPISODE54_DATASET=/path/to/A/121_430_data_lerobot
export EPISODE54_ALIGNMENT=/path/to/wrist_camera_alignment_20261103
export EPISODE54_WRIST_JSON=/path/to/wrist_camera_calibration_palm_left.json
export EPISODE54_REPLAY_H5=/path/to/private-working-directory/episode54_joint_replay.h5

"$ISAAC_PYTHON" scripts/build_real_joint_replay.py \
  --dataset "$EPISODE54_DATASET" --episode 54 --output "$EPISODE54_REPLAY_H5"
"$ISAAC_PYTHON" episode54-real-input/run_episode54_replay.py
"$ISAAC_PYTHON" episode54-real-input/run_episode54_new_material_replay.py
```

回放还需要 OpenCV 和 FFmpeg。脚本以时间戳另建 `reruns/` 或 `material_runs/`。其中可能包含真机画面，按混合来源归档。重新导出混合分析时，调用 `python episode54-real-input/build_deliverables.py PATH_TO_COMPLETED_RERUN`，输出到该次运行的 `mixed-analysis/`。两个历史实验分别进行了 3,804 步回放。刚体版本验证了轨迹回放。原生变形体版本产生 951 帧果实节点输出，出现明显过大的 85.5 mm 最大节点位移和 302.27 N 指尖峰值，适合定位接触与材料问题。新增参数化入口本次未用真机原始输入重跑。

[单果诊断](single-fruit-real-pose/progress_report.md)从 Episode 54 首帧姿态出发，使用独立的番茄位置与闭合几何。在 `single-fruit-real-pose/` 中提供了固定果实的 FEM / 凝胶代理视频和逐帧结果。重新生成可视化或代理图前将实验目录复制到可写工作区，脚本会更新同名输出。重跑闭合探测时用 `--config` 传入授权工作区的 `selected_initial_state.json`，该文件含实测首帧姿态。TacMap 代理脚本另需用 `SHARPA_TACMAP_ASSETS` 指明官方 `tactilesensor_map` 资产路径。
