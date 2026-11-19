# 双 FR3 / Sharpa 番茄采摘仿真交接

新同事先跑一条 [Newton 启动样本](#1-newton-主仿真)，再按[验收步骤](#3-输出与验收)查看录制和质量报告。

| 位置 | 内容 |
| --- | --- |
| [tomato-pick-sim/](tomato-pick-sim/) | 独立 Newton 工程、模型资产、十种场景配置、锁定依赖、原始文档和测试 |
| [isaac-lab/](isaac-lab/) | HandsOn 扩展、Isaac Sim 场景构建、策略客户端、使用真机输入的独立回放实验 |
| [material-studies/](material-studies/) | 番茄外部 FEM、Isaac 压缩、Sharpa 局部凝胶及 TacMap |
| [evidence/newton/](evidence/newton/) | 纯仿真的启动、历史脚本采集及 OpenPI 质量报告 |
| [下载录像与 HDF5](https://github.com/ZoeYwooD-Li/trex-sharpa-tomato/releases/tag/tomato-simulation-20261119) | 纯仿真的历史 Newton Rerun 和 OpenPI 记录，附 SHA-256 |

根目录的 `trex_sharpa_adapter/`、训练缓存和传输脚本属于后训练和真机接口流程。复现仿真时执行本目录的入口。Episode 54 原始轨迹、RGB-D、触觉、标定和带真机画面的拼接视频由团队原始资料渠道管理。Episode 300 三相机触觉视频属于真机采集复核。

## 1. Newton 主仿真

准备 Linux、Python 3.11、uv 和能运行 CUDA 13.0 PyTorch 构建的 NVIDIA 驱动。已有验证环境为 RTX 5090、Newton 1.5.0、Warp 1.16.0、MuJoCo 与 MuJoCo Warp 3.11.0。本工程、旧训练环境和 Isaac 各自使用独立的 Python 环境。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ZoeYwooD-Li/trex-sharpa-tomato.git
cd trex-sharpa-tomato/simulation/tomato-pick-sim
uv sync --locked
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest

uv run --locked tomato-sim run \
  --config configs/cherry_tomato_pick_sharpa.yaml \
  --frontend headless --duration 0.1
```

`GIT_LFS_SKIP_SMUDGE=1` 跳过根目录的后训练大文件。仿真资产随 Git 目录提供。短跑应完成 20 个物理步，写出 `outputs/cherry_tomato_pick_sharpa/episode_*.h5` 及其 `.quality.json`。短跑的任务尚未完成，预期 `rejected`，进程可能因任务验收返回 1。确认模型、三路相机和录制成功初始化后，再运行完整尝试。

```bash
uv run --locked tomato-sim run \
  --config configs/cherry_tomato_pick_sharpa.yaml --frontend rerun --live

uv run --locked tomato-sim batch \
  --config configs/cherry_tomato_pick_sharpa.yaml --attempts 3 --seed 42

uv run --locked tomato-sim quality \
  outputs/cherry_tomato_pick_sharpa/BATCH_DIRECTORY --summary
```

以实际生成的批次目录替换 `BATCH_DIRECTORY`。默认一次尝试最多 28 秒仿真时间，物理步长 0.005 秒。脚本控制中，右手支撑果串，左手摘果再投入目标盒。目标、摆位及外观可随机化。果实脱离并在指定盒内稳定约 0.3 秒时，系统补采末帧并提前收尾。只看画面可加 `--set recording.enabled=false`。

## 2. 配置、代码与策略入口

| 要修改的内容 | 入口 |
| --- | --- |
| Sharpa 采摘、随机化、物理、相机或录制 | [cherry_tomato_pick_sharpa.yaml](tomato-pick-sim/configs/cherry_tomato_pick_sharpa.yaml) |
| 接近、支撑、摘果、投放和提前停止 | [scene.py](tomato-pick-sim/demos/cherry_tomato_pick/scene.py) |
| 接触、可断果梗和脱离 | [backend.py](tomato-pick-sim/demos/cherry_tomato_pick/backend.py) |
| 双手模型、安装变换与 FK/IK | [sharpa.py](tomato-pick-sim/demos/cherry_tomato_pick/sharpa.py)、[kinematics.py](tomato-pick-sim/demos/cherry_tomato_pick/kinematics.py) |
| HDF5 和质量门槛 | [recording.py](tomato-pick-sim/demos/cherry_tomato_pick/recording.py)、[quality.py](tomato-pick-sim/demos/cherry_tomato_pick/quality.py) |
| 历史 Wuji2、原采集基线、主梗夹持、果梗朝上及包络抓持 | [配置](tomato-pick-sim/configs/)和[实验说明](tomato-pick-sim/docs/) |

历史 Wuji2 使用每手 20 个关节。Sharpa 使用每手 22 个，两臂两手位置按 7+22+7+22 排列，共 58 维。两者的动作宽度都可能是 16，切换手型仍须核对动作语义和安装坐标。运行历史配置时显式传 `--config`。

OpenPI 策略进程和 Newton 进程各用各的环境。先按 [OpenPI 说明](tomato-pick-sim/docs/openpi.md)启动与权重匹配的模型服务，再在 Newton 工程运行下列命令。权重由团队渠道提供。

```bash
uv sync --locked --extra openpi
uv run --locked --extra openpi tomato-sim eval \
  --policy-url ws://127.0.0.1:8000 --checkpoint-label YOUR_CHECKPOINT \
  --execute-steps 4 --duration 28 \
  --set recording.seed=42 --set recording.vary_seed_by_episode=false
```

检查服务端的动作合同。当前 Newton 的 OpenPI 接口读取三路 200×320 RGB 和 16 维状态，执行世界坐标下的绝对 TCP 位姿及闭合量。历史 `openpi_delta_eval` 的控制模式以各自 `.evaluation.json` 为准。T-Rex 的 Isaac 闭环使用另外的客户端、状态和坐标约定，见 [Isaac 交接](isaac-lab/README.md)。

## 3. 输出与验收

1. 核对 HDF5 和同编号的 `.quality.json`。策略尝试另有 `.evaluation.json`。核对控制模式、种子、目标盒、结束原因和执行命令。
2. 读取记录内的时间戳。机器人和腕相机默认约 30 Hz，头相机约 10 Hz，提前结束的末帧可能落在常规采样点之间。Sharpa 关节位置宽度为 58，TCP 动作宽度为 16。
3. 将 `accepted: true` 和任务终止原因一起核对。`accepted.txt` 是批次接受索引，通过率的分母用实际尝试次数。启动、超时、仅接近、力过大或投错盒子的样本用于排查问题。
4. 本次 266 项测试通过。RTX 5090 的 0.1 秒短跑写出 3 个机器人样本，三路相机各有 1、3、3 帧，按预期[被质量门槛拒绝](evidence/newton/handoff_startup.quality.json)。历史 OpenPI 的 [28 秒样本](evidence/newton/openpi_delta_eval/full_eval/episode_000001.quality.json)也被拒绝，原因包括采摘投放未完成及右手支撑未就绪。
5. 当前证据覆盖启动、录制、策略接口及失败诊断。完整采摘通过率仍需用固定种子和目标盒组合测量。

## 4. 历史资料与文件管理

- Newton 来源为独立工程提交 `a17d84d15844afc93690707845be231e911d2678`。保留[提取来源与许可证](tomato-pick-sim/provenance/README.md)、[Sharpa 集成说明](tomato-pick-sim/docs/sharpa.md)及资产许可证。两份历史 LycheeSim refresh 工作区指向同一提交 `510b102`，日常开发使用本次归档的独立工程。
- 下载两份 Release 压缩包及 `SHA256SUMS` 后运行 `sha256sum -c SHA256SUMS`。解压后可从解压根目录运行 `sha256sum -c SIM-RECORDINGS-SHA256SUMS` 检查各原始文件。Rerun 预览可在 Newton 目录执行 `uv run --locked rerun /path/to/preview_1s.rrd`。
- 源码、配置、资产和报告放 Git，大体积 HDF5 与 RRD 放 Release。运行中的产物写入 `outputs/`。归档新实验时登记日期、提交、环境、配置、种子、数据来源和质量状态。
- Isaac 的重复网格导出、旧相机方案、全量逐帧缓存和参数搜索产物仍在原历史工作区。一般性的 Hydroshear 触觉比较属于独立项目。
