# T-Rex × 双 FR3 / Sharpa：全参数权重推理交接

本目录提供 **GPU 侧模型服务** 的安装补丁、启动和无机械臂探针。它不包含真机 SDK 客户端、机器人控制程序、数据集或权重；机械臂运动需在独立部署代码中实现并验收。

## 1. 代码与权重

GPU 机器使用与训练匹配的官方 T-Rex 基准提交 `ba4047ce3e9f3d2013d60de600e136897e90ccb8`，加本仓库的三个安装器：

```bash
git clone https://github.com/ZhuoyangLiu2005/T-Rex.git T-Rex
git -C T-Rex checkout ba4047ce3e9f3d2013d60de600e136897e90ccb8
git clone --branch full-finetune-server \
  git@github.com:ZoeYwooD-Li/trex-sharpa-tomato.git trex-sharpa-tomato
python3 trex-sharpa-tomato/trex_sharpa_adapter/install_adapter.py --repo T-Rex
python3 trex-sharpa-tomato/trex_sharpa_adapter/install_lora.py --repo T-Rex
python3 trex-sharpa-tomato/trex_sharpa_adapter/install_inference.py --repo T-Rex
```

`install_lora.py` 也安装全参数训练时使用的模型修正；推理完整权重时不注入 LoRA。安装器仅对固定基准源码应用可核对补丁，遇到未识别的代码会报错。**训练服务器当时的完整源码状态仍应与此结果比较**；若存在额外未记录补丁，先核对再加载实机权重。

只把选中的完整 `checkpoint-0-N/` 目录传到 GPU 机器。目录至少包含 `model.pt`、`config.json`、`training_args.json`、`stats_data.json`、`adapter_metadata.json`、`processor/` 和 `robot/`。原始训练数据、`prepared_sharpa/`、midtrain 权重均不是完整全参数 checkpoint 推理的必需输入。不要把 checkpoint 提交到 Git。

本仓库的脚本用 `PYTHON_BIN` 指定已安装 T-Rex 依赖的 Python。需在目标 GPU 上确认 CUDA 可用，且环境有与训练兼容的 Qwen3-VL `transformers`、PyTorch、`pyzmq`、Pillow、NumPy 等依赖。此前训练环境用 `transformers==4.57.3`、`huggingface-hub==0.36.2`；不要把 π0.5/JAX 公共环境当作 T-Rex 环境。

## 2. 启动与无机械臂测试

在 GPU 机器上：

```bash
export TREX_ROOT=/path/to/T-Rex
export CHECKPOINT=/path/to/checkpoint-0-6250
export PYTHON_BIN=/path/to/trex-env/bin/python
export CUDA_DEVICE=0 PORT=5678 BIND_HOST=127.0.0.1
bash trex-sharpa-tomato/trex_sharpa_adapter/serve_sharpa.sh
```

脚本先检查权重目录、62D/16 帧动作、十指触觉、相机身份和 link8 URDF 哈希，然后加载模型并做一次零值预热。服务端使用 ZMQ REP；出现 `Warm-up output shape: (16, 62)` 和监听日志才表示启动成功。另一终端用同一环境运行：

```bash
/path/to/trex-env/bin/python \
  trex-sharpa-tomato/trex_sharpa_adapter/probe_server.py --port 5678
```

探针只发送合成图像和触觉，不连接机械臂；通过后应打印 `(16, 62)` 和有限值检查结果。模型加载或探针通过不代表真机效果已验证。

服务默认只绑定 `127.0.0.1`。当 GPU 机器与机器人主机分离时，在**机器人主机**建立 SSH 端口转发，然后客户端连接其本机 `127.0.0.1:5678`：

```bash
ssh -N -L 127.0.0.1:5678:127.0.0.1:5678 <gpu-user>@<gpu-host>
```

当前协议反序列化 Python pickle，只允许可信进程访问；不要将服务端口直接暴露到不可信网络。

## 3. 真机客户端必须满足的合同

- 三路图像键为 `image_head`、`image_wrist_right`、`image_wrist_left`，对应训练数据 cam0、cam1、cam2；值为编码后的 RGB 图像字节。任务文本键为 `task_description`，使用与训练任务一致的英文描述。
- 触觉键为 `tactile_f6`，原始值形状 `[10,6]` 或历史窗口 `[16,10,6]`，左手五指后右手五指，顺序为 thumb/index/middle/ring/pinky，各指 Fx/Fy/Fz/Tx/Ty/Tz；不要重复乘倍率。`tactile_deform` 为 `[10,240,240]` 的 uint8 或 0–1 浮点图，顺序相同。当前 checkpoint **需要触觉**。
- chunk 开始发送 `mode="slow_and_fast"`，之后可在 chunk 内发送 `mode="fast"` 更新触觉；每次响应检查 `status="success"`、`chunk_id`、有限值和动作形状 `[16,62]`。官方参考 cadence 为 chunk 起点及第 4、8、12 帧更新。重复使用旧 chunk、网络超时或传感器缺帧时应停止本轮执行，不能继续下发旧动作。
- 每帧 62D 为：左臂 `xyz + rot6d` 9、左手绝对关节 22、右臂 `xyz + rot6d` 9、右手绝对关节 22。两个臂的相对位姿均以**该 chunk 开始时各自实测 link8 法兰位姿**为基准，平移在起点法兰坐标系；rot6d 为相对旋转。需先组合目标法兰位姿，再由匹配 link8 的 IK 和现场 Safety Gateway 变为关节指令。手指单位为 rad，直接作为绝对目标。不可把这 62D 送入旧 π0.5 的 58D 关节动作入口，也不可把每步当作相邻帧差分。
- 真机侧先核对相机、触觉、当前关节、IK 参考点与现场标定；先只读观测和不下发动作的推理记录，再低速、单 chunk 测试。当前代码包不实施机械臂控制。

## 4. 本次修正

`install_inference.py` 对论文版 `scripts/test.py` 做四件事：以 `[10,6]` 和十指形变预热双臂模型；权重键缺失或多出时停止启动；只加载权重张量并使用 mmap；新增 `--bind_host`，默认本机回环地址。它不更改训练结果。真机客户端仍需在部署仓库单独适配。
