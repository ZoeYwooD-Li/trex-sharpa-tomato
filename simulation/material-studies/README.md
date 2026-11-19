# 番茄材料、接触与触觉实验

| 实验 | 入口与结果 |
| --- | --- |
| 樱桃番茄 SciPy 外部 FEM | [求解器和参数](cherry_tomato_material_sim/)及[验证报告](cherry_tomato_material_sim/validation_report.md)，包括四面体网格、参数灵敏度、力位移曲线和 USD |
| Isaac 番茄压缩对照 | [压缩说明](isaac_tomato_compression_README.md)、[示例输出](isaac_tomato_gpu_deliverable/)及诊断脚本 |
| Sharpa 4F 局部接触和凝胶 | `sharpa_local_contact_fem.py`、`sharpa_local_tacmap.py`、`sharpa_deformed_mesh_probe.py` 及[官方数据审计](sharpa_official_data_audit.md) |
| 固定果实与触觉代理 | [实验过程和数值边界](../isaac-lab/single-fruit-real-pose/progress_report.md)。依赖归档的闭合几何或团队侧真机首帧姿态，归在 Isaac 混合来源目录 |

纯数值基线使用 Python、NumPy、SciPy、Trimesh 和 USD Python 模块。先创建独立环境，安装 `numpy scipy trimesh usd-core`，再从本目录运行：

```bash
python cherry_tomato_material_sim/run_cherry_tomato_fem.py \
  --output "$PWD/outputs/cherry_tomato_fem_$(date +%Y%m%d_%H%M%S)"
```

需要查看接触模型时，核对网格、反力、边界载荷、形变、单位和四面体反转，并比较不同网格密度。候选参数为杨氏模量 281 kPa、泊松比 0.335、果实质量 6.644 g，尚待真实压缩试验标定。局部凝胶模型和 TacMap 代理存在载荷范围与映射假设。原始资料保留作者标注和实验过程，带旧路径的历史诊断脚本需要按本机依赖调整。本次只验证主仿真与普通 Isaac 启动，材料结果沿用历史记录。
