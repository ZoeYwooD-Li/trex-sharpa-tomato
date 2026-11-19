"""Render nominal open/closed Wuji2 mesh poses; not a physics-success test."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from demos.cherry_tomato_pick.backend import load_optional_visual_assets
from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot


def main() -> None:
    root = Path.cwd()
    cfg = yaml.safe_load((root / "configs/cherry_tomato_pick_main_stem.yaml").read_text())
    target = np.asarray(cfg["scene"]["tomato_position"])
    center = target + cfg["scene"]["stem_grasp_offset_m"]
    robot = DualWuji2PickRobot(
        cfg["robot"], repository_root=root,
        tomato_position=target, support_position=center,
    )
    visuals = load_optional_visual_assets(root, cfg)
    fig = plt.figure(figsize=(12, 10))
    for column, closure in enumerate((0.0, 1.0)):
        robot.set_targets(robot.left_grasp_tcp, robot.right_grasp_tcp, 0.0, closure)
        objects = []
        for part in robot.visual_parts:
            if not part.name.startswith("robot/right_wuji2/"):
                continue
            position, quat = robot.visual_pose(part)
            vertices = Rotation.from_quat(quat).apply(part.vertices) + position
            # Keep full authored geometry for finger silhouettes.
            objects.append((vertices, part.faces, "#737e8a"))
        for name, visual in visuals.items():
            origin = target.copy()
            if visual.body_name == "support_tomato":
                origin += cfg["scene"]["support_tomato_offset_m"]
            vertices = visual.vertices + visual.local_position + origin
            color = "#347c35" if "stem" in name or name == "branch" else "#d92b26"
            objects.append((vertices, visual.faces, color))
        for row, (elev, azim) in enumerate(((80, 0), (20, 5))):
            ax = fig.add_subplot(2, 2, row * 2 + column + 1, projection="3d")
            for vertices, faces, color in objects:
                ax.add_collection3d(Poly3DCollection(
                    vertices[faces], facecolors=color, linewidths=0, shade=True,
                ))
            ax.scatter(*center, color="#00c5df", s=35, depthshade=False)
            ax.scatter(*target, color="#ffc400", s=35, depthshade=False)
            ax.set_xlim(center[0] - .12, center[0] + .15)
            ax.set_ylim(center[1] - .135, center[1] + .135)
            ax.set_zlim(center[2] - .06, center[2] + .21)
            ax.set_box_aspect((1, 1, 1), zoom=1.45)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            ax.set_title("Reached TCP: OPEN" if closure == 0 else "Nominal five-finger WRAP")
    fig.suptitle("Middle-stem grasp | cyan: stem centre; gold: harvest target\nNominal mesh poses, not measured physics success")
    fig.tight_layout()
    output = root / "outputs/middle_stem_grasp/nominal_hand_poses.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
