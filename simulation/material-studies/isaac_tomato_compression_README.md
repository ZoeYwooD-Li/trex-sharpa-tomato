# 单颗小番茄压缩：PhysX 审计与 FEM 基线

原采摘场景没有修改。当前可复现的受力结果由独立的准静态四面体有限元求解器计算；导出的 USD 带有计算节点位置的时间采样，可在 Isaac Sim 打开播放。播放时是在显示已求解的形状序列，接触力来自 FEM 求解，不是 PhysX 的实时接触反力，也不是手工改渲染网格。

## 原 PhysX 场景审计

| 项目 | 原实现中生效的设置 | 审计结果 |
| --- | --- | --- |
| 引擎 | Isaac Sim 5.1 / PhysX volume deformable，GPU pipeline | 已在 RTX 5090 上启动验证 |
| 单位 | `metersPerUnit = 1`；`kilogramsPerUnit` 未显式写入，USD 默认 1 kg/unit | 坐标是 m；`omniphysics:mass` 是 kg；杨氏模量按 Pa 写入 |
| 番茄 | 球半径 0.0172 m，质量 0.020 kg；初始底点距地 0.0005 m | 来自旧项目配置，未用实物测量校准 |
| 软体 | 体积 deformable，E=800,000 Pa，ν=0.42，damping=0.25 | E 沿用原 Isaac 原型的试值；ν、阻尼未校准 |
| 网格 | cooking surface 为 subdivision-2 icosphere；163 个仿真节点；旧场景将仿真 TetMesh 同时用作接触网格 | 两种网格职责混在一起，碰撞分辨率受仿真网格限制 |
| 地面 | 尺寸 1 m 立方体，缩放到 0.5×0.5×0.1 m；顶面 z=0；有 `UsdPhysics.CollisionAPI` | 是静态刚性碰撞体；没有被隐藏或移动；未发现碰撞过滤关系 |
| PhysX 求解器 | TGS、GPU broadphase、120 Hz、位置迭代上限 32；软体位置迭代 32 | 时间步长 1/120 s |
| 软体碰撞偏移 | 旧场景 contactOffset=0.001 m，restOffset=0 m | 实际约 1 mm 接触偏移；不是把地面挪开 |

PhysX 短测确实运行，但较长的 0.2 N 等效重力测试出现约 50% 以上的高度缩短和毫米级地面穿透。把接触 TetMesh 与仿真 TetMesh 分开后，PhysX GPU tensor view 又报 `Unable to copy data from host to device for collElementIndicesBuffer`，因此旧 PhysX 结果不作为校准数据。旧场景与诊断脚本分别保存在 `small_tomato_ground_physx_diagnostic_failed.usda`、`isaac_tomato_physx_diagnostic_failed.py`。NVIDIA 文档也列出 deformable body 不支持 contact reports，GPU deformable 接触无法从普通 rigid contact report 直接取得反力；见 [Isaac Sim 5.0 PhysX limitations](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/physics/physics_resources.html)。

## 当前 FEM 模型

- 一个四面体实体番茄、一个固定地面和一个刚性平板。地面顶面 z=0，初始最底节点 z=0，没有初始穿透。
- 用 Delaunay 生成 698 节点、1,958 四面体和 1,280 个外表面三角形。地面/平板接触通过单边罚函数求解；刚性板位移直接施加到求出的接触节点，反力由 FEM 约束反力计算。
- 采用均匀、各向同性、小应变线弹性材料；重力静置后，平板位移到 0.5、1、2 mm，并求平衡。接触迭代收敛后记录反力。
- 罚刚度为 1e7 N/m。接触面积是接触三角面的投影近似值，网格质量用 0–1 四面体形状质量比报告，1 为理想正四面体。
- 不包含黏弹性、果皮/果肉分层、破裂或大变形超弹性；本轮是稳定单材料基线，不是番茄材料标定。没有实物压缩数据时，不把假设参数称为测量值。

### 参数来源

| 参数 | 值 | 来源/状态 |
| --- | ---: | --- |
| 半径 | 17.2 mm | 旧项目配置；未测量校准 |
| 质量 | 20 g | 旧项目配置；未称重校准 |
| 密度 | 946.48 kg/m³ | 由 FEM 网格实际体积与 20 g 推导 |
| 泊松比 | 0.42 | 假设；未标定 |
| 杨氏模量 | 0.2、0.8、2 MPa | 三组假设试值；0.8 MPa 沿用旧原型值 |
| 重力 | 9.81 m/s² | SI 标准值 |
| 接触罚刚度 | 1e7 N/m | 数值求解参数；不是水果属性 |

## 基线结果（E=0.8 MPa）

重力静置：最低节点 z=0 m，地面穿透 0 m；高度从 34.400 mm 变为 34.361 mm（缩短 0.039 mm）；体积变化 −0.00318%；四面体最小质量比从 0.26452 变为 0.26448；倒置单元 0。

| 平板行程 | 番茄高度缩短 | 反力 | 压缩率 | 接触面积近似 | 体积变化 | 最大地面穿透 | 最小四面体质量比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 mm | 0.500 mm | 1.98 N | 1.45% | 34.3 mm² | −0.0290% | 0 m | 0.2641 |
| 1 mm | 1.000 mm | 7.36 N | 2.91% | 68.6 mm² | −0.0589% | 0 m | 0.2638 |
| 2 mm | 2.000 mm | 23.15 N | 5.82% | 162.2 mm² | −0.1705% | 0.000044 mm | 0.2622 |

2 mm 时没有倒置单元。网格接触面积是有限网格上的几何近似，结果随网格密度和接触罚刚度变化；不能替代实际压盘测量。

三组刚度下的平板反力（N）：

| 杨氏模量 | 0.5 mm | 1 mm | 2 mm |
| ---: | ---: | ---: | ---: |
| 0.2 MPa | 1.51 | 1.85 | 5.81 |
| 0.8 MPa | 1.98 | 7.36 | 23.15 |
| 2 MPa | 4.94 | 18.41 | 57.80 |

## 运行与查看

在当前环境复现 FEM 解、CSV、图和 USD：

```bash
/path/to/local/anaconda3/envs/RoboDojo/bin/python outputs/tomato_fem_compression.py --show-meshes --output outputs/tomato_fem_results
```

用 Isaac Sim 打开 `outputs/tomato_fem_results/tomato_fem_playback.usda`，播放时间轴查看重力静置以及 0.5、1、2 mm 的平板压缩状态。默认 USD 显示黄色外表面接触网格和蓝色四面体边线。脚本、CSV、配置/网格验证、图分别为：`tomato_fem_compression.py`、`force_displacement.csv`、`validation.json`、`fem_force_displacement.png`、`fem_deformation_comparison.png`。

这个 USD 是 FEM 平衡态结果的播放文件，打开后按 Isaac Sim Play 只是播放计算好的结果序列；若要改材料或网格，需重新运行 FEM 脚本。渲染播放期间没有 PhysX 实时 contact force。
