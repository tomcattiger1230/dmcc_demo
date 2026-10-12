# dmcc_demo

This is a demo code for our TR-O paper "Time-Optimal Handover Trajectory Planning for Aerial Manipulators based on Discrete Mechanics and Complementarity Constraints"

本仓库是带单自由度机械臂的 7 自由度 DMCC 演示。修补后的默认命令采用
一致的 Euler 坐标动力学和纯时间目标；原场景位置保留。

使用 uv 安装和运行：

```bash
uv sync --locked
uv run --locked python uav_mani_DMCC_demo.py --task drone_racing --no-plot
uv run --locked python uav_mani_DMCC_demo.py --task fixed_point_contact --no-plot
uv run --locked python -m unittest discover -s tests -v
```

移除 `--no-plot` 显示原有三种轨迹图。结果和模型模式保存在 `results/*.npy`、
`results/*.json`；目录自动创建，失败或违反约束的解不会作为成功结果导出。
`--nodes 41` 指定节点数，`--pass-speed 0.1` 指定中间航点的最小离散区间速度（m/s）。

历史角速度/力矩及离散力约定可显式选择：

```bash
uv run --locked python uav_mani_DMCC_demo.py --task drone_racing \
  --dynamics-convention legacy --objective regularized --no-plot
```

时间步长、航点顺序、参考保护等修补统一生效，legacy 并不复现原来的错误。
源码修补前版本为 `e56d160dc85b406c7557ac6dd8a25d5c2590c214`。
模型改动、容差单位、模式边界和实际验证见 [修补说明](docs/formulation_repairs.md)。
