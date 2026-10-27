# T-Rex × 双 FR3 / Sharpa 番茄采摘：后训练接入包

本仓库保存适配器源码和已经生成的数值缓存。原始四个 LeRobot v2.1 数据集（Parquet、RGB 视频、meta，约 44 GiB）需单独传到训练服务器；缓存不能代替原始数据。仓库不包含 T-Rex 官方代码或模型权重。

## 内容

- `trex_sharpa_adapter/`：适配器源码、FR3 名义模型、检查报告及训练脚本。
- `trex_sharpa_adapter/INFERENCE.md`：全参数及已合并 LoRA checkpoint 的 GPU 推理启动、无机械臂探针与真机接口合同。
- `trex_sharpa_adapter_v3.zip`：Windows 交接时的原始 v3 包。解压目录中的 `relocate_prepared.py` 另有一个 Linux 依赖修复：重定位只读 JSON，不再因导入 `sharpa_core` 要求预先安装 PyArrow。
- `prepared_sharpa_link8.zip`：A+B 的数值缓存与训练集统计，使用 Git LFS 管理。

原始 ZIP 的 SHA256：

```text
8b9111c5888e84a0947a9056551c10d2f0b2235145e447f7437cbee0538699e0  trex_sharpa_adapter_v3.zip
c2cb7894a5926152dcd674d4da28eba8f293cc840535f2e3eb6c806ec4e28330  prepared_sharpa_link8.zip
```

## 在训练服务器准备

先在训练服务器配置对这个**私有仓库**的读取权限并安装 Git LFS，然后：

```bash
git clone https://github.com/ZoeYwooD-Li/trex-sharpa-tomato.git
cd trex-sharpa-tomato
git lfs pull
sha256sum -c <(printf '%s  %s\n' c2cb7894a5926152dcd674d4da28eba8f293cc840535f2e3eb6c806ec4e28330 prepared_sharpa_link8.zip)
```

确认缓存 ZIP 的实际大小约为 221 MiB，而非几百字节的 LFS 指针。把原始四个数据集放在同一个数据根目录，布局如下：

```text
/data/sharpa/
  A/121_430_data_lerobot/{data,videos,meta,...}
  A/54_120_data_lerobot/{data,videos,meta,...}
  B/130_423_data_lerobot/{data,videos,meta,...}
  B/56_106_data_lerobot/{data,videos,meta,...}
```

源数据的 A/54 和 B/56 是符号链接；传输时需要跟随链接复制内容，例如使用 `rsync -aL`。训练时建议读取服务器本地磁盘，不要依赖不稳定的网络挂载。

仓库提供 `transfer_raw_data.sh`。从能访问原始数据的机器运行，先在服务器创建目标目录，例如：

```bash
export SOURCE_ROOT=/path/to/data-lerobot/sharpa
export DEST_ROOT=user@training-server:/data/sharpa/
bash transfer_raw_data.sh
```

脚本只传这四个数据集，且会跟随 A/54 与 B/56 的符号链接；服务器上需要已存在 `/data/sharpa/`。传完后检查四个目录均有 `data/`、`videos/`、`meta/`，再运行下方的缓存重定位。

```bash
mkdir -p /workspace/trex_job
git clone https://github.com/ZhuoyangLiu2005/T-Rex.git /workspace/trex_job/T-Rex
git -C /workspace/trex_job/T-Rex checkout ba4047ce3e9f3d2013d60de600e136897e90ccb8
unzip prepared_sharpa_link8.zip -d /workspace/trex_job
python3 trex_sharpa_adapter/install_adapter.py --repo /workspace/trex_job/T-Rex
python3 trex_sharpa_adapter/relocate_prepared.py \
  --prepared /workspace/trex_job/prepared_sharpa --data-root /data/sharpa
```

若使用单卡 LoRA，还需执行 `python3 trex_sharpa_adapter/install_lora.py --repo /workspace/trex_job/T-Rex`，再按 [LoRA 步骤](trex_sharpa_adapter/README.md#可选单-gpu-lora-后训练)运行 `train_sharpa_lora.sh`。LoRA checkpoint 只保存适配参数；部署时用 `merge_lora.py` 与同一 midtrain 权重合并。

本仓库已有 LoRA 训练与合并代码。部署时可使用全参数 `checkpoint-0-N/`，或使用**已合并**的 LoRA 完整 checkpoint；两者共用 `serve_sharpa.sh`。原始 `lora.pt` 目录不能直接作为服务权重，具体步骤见 [推理交接](trex_sharpa_adapter/INFERENCE.md)。

8 卡全参数后训练也应执行 `install_lora.py`，但使用新增的 `train_sharpa_full.sh`（强制 `--lora_rank 0`）；该安装器同时提供按步保存和多卡学习率步数修正。先用默认 20 batch 短跑，再启动新的正式输出目录。命令及训练语义见[全参数后训练步骤](trex_sharpa_adapter/README.md#8-卡全参数后训练rtx-pro-5000-blackwell)。

本地 RTX 5090 已用完整 midtrain checkpoint、三路 RGB、十指触觉和 FLARE 跑通 20 个 LoRA 训练 batch 与两次验证，结果和适用范围记录在 [短跑报告](trex_sharpa_adapter/lora_smoke_5090.json)。服务器上的正式实验仍需把原始四个 LeRobot 数据集复制到本地磁盘，并按服务器 GPU 重新验证环境。

适配器目录可以保留在本仓库；运行 `train_sharpa.sh` 时设置 `TREX_ROOT`、`PREPARED_ROOT`、`ORIGIN_MODEL_PATH`、`RESUME_CHECKPOINT`、`OUTPUT_DIR`、`NUM_PROCESSES`、`ACCELERATE_CONFIG`、`TRAIN_BSZ`。基础模型是 `Qwen/Qwen3-VL-2B-Instruct`；midtrain 权重是 `miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6`，交接时核对的 revision 为 `62efb3bcb45a3df0e088c8909d759b582cfb98af`。保持 `--resume_source midtrain`。完整验证与 20 batch 短跑步骤见适配器的 `README.md` 和 `train_sharpa.sh`。

训练环境须按服务器 GPU 选择。官方固定的 Torch 2.6.0/cu124 不支持本地 RTX 5090（sm_120）；若服务器也是 5090，需要先解决 PyTorch/CUDA 与 DeepSpeed 兼容性，不能直接照抄官方安装命令。
