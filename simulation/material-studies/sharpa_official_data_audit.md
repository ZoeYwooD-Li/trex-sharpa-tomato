# Sharpa TacMap 官方数据核对

核对对象：`work/sharpa_tacmap_official` 浅克隆，官方仓库 `sharpa-robotics/sharpa-tacmap`；截至 2026-11-07。该克隆仅用于只读检查。

## 仓库中确实提供的数据

- Wave 手模型、圆柱和平板压头 USD。
- 4F/拇指传感器采样表面点与法线 `.npy`，供 TacMap 射线/接触图计算。
- 两个压入案例 JSON 与目标轨迹 `.npy`，提供几何/位姿动作序列。
- 没有在仓库中找到配对的真实指尖触觉帧与外部力传感器逐帧读数。

## 官方示例里的力是什么

`env.py` 的 `compute_observations()` 从 Isaac Lab `ContactSensor.data.net_forces_w_history` 取出接触体的净接触力，并把第 0 帧历史值放入 `tactile_forces`。`tactile_align_wrapper.py` 的 `--save_exit` 会把运行时结果存成 `sim_tactile_force.npy`。该文件是运行仿真后生成的输出，不是仓库随附的真实测量标签。它适合复现官方仿真流程与检查信号格式；其物理可信度仍需与独立力平衡或公开试验数据对照。

## 论文公开结果能支持什么

论文说明真实力标签由外部高精度力传感器采集，并报告模拟与实测力的中位 L2 误差：方块 0.28 N、圆柱 0.61 N。当前可查到论文的汇总指标和图，没有逐帧原始力/触觉序列可下载。因此这些指标可以作为外部参考，不能恢复出监督训练集，也不能单独重算力—位移曲线。

## 对当前验证的用法

1. 官方点/法线继续用于检查 TacMap 几何映射。
2. 官方轨迹可以作为复现动作输入；仿真生成的 `sim_tactile_force.npy` 可用于核对官方接触传感器信号链。
3. 当前准静态接触反力继续用独立 FEM 约束反力，并以 Ecoflex 实测压缩曲线校验材料力学。
4. 论文力误差仅列为参考指标。拿到官方逐帧外部力数据或本地硬件标定数据前，不将其宣称为触觉图到真实力的监督验证。

## 本地复现结果

Isaac Sim 5.1 / Isaac Lab GPU headless 下，已用仓库两组官方案例各运行一遍。官方仓库原始入口的导入路径与目录布局不匹配；为复现仅在本地克隆副本调整了 `run.py` 的包导入，未改上游项目或用户原始场景。两个案例均走完并保存结果。

| 案例 | 保存帧数 | 峰值接触力模长 | 平均力模长 | 有力帧 | 有 TacMap 深度像素比例均值 |
|---|---:|---:|---:|---:|---:|
| cylinder_D4 | 100 | 4.125 N | 1.673 N | 89 | 1.85% |
| square_4 | 150 | 4.435 N | 1.408 N | 134 | 1.59% |

帧记录的力是 Isaac `ContactSensor` 的世界坐标净接触力向量；TacMap 是官方 `uint8` 分段量化输出。数值证明官方仿真链路可在本环境跑完并同时产生力与深度输出，但这两个量来自各自的仿真模块，尚未与公开的真实逐帧力标签比较。逐帧数据见 `outputs/sharpa_official_replay/force_trace.csv`，汇总见 `outputs/sharpa_official_replay/summary.json`。

官方来源：

- [TacMap GitHub 仓库](https://github.com/sharpa-robotics/sharpa-tacmap)
- [TacMap 论文](https://arxiv.org/abs/2602.21625)
