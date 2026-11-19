"""Photo-inspired procedural tableware; metres, world XYZ, open convex compounds."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


@dataclass
class Fixture:
    name: str
    mesh: trimesh.Trimesh
    colliders: list[trimesh.Trimesh]
    color: tuple[float, float, float]


def _box(size, center):
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(center)
    return mesh


class Tableware:
    def __init__(self, cfg):
        self.cfg = cfg
        self.plate_center = np.asarray(cfg["plate_center_xy"], dtype=float)
        self.plate_radii = np.asarray(cfg["plate_radii_m"], dtype=float)
        self.centers = np.asarray(cfg["box_centers_xy"], dtype=float)
        self.width = float(cfg["box_width_m"])
        self.wall = float(cfg["wall_thickness_m"])
        self.floor = float(cfg["floor_top_z_m"])
        self.height = float(cfg["box_height_m"])
        self.plate_height = float(cfg["plate_rim_height_m"])
        if (self.plate_center.shape != (2,) or self.plate_radii.shape != (2,)
                or self.centers.shape != (3, 2)):
            raise ValueError("tableware requires one XY plate center/radii and three XY box centers")
        values = np.r_[self.plate_center, self.plate_radii, self.centers.ravel(),
                       self.width, self.wall, self.floor, self.height, self.plate_height]
        if (not np.all(np.isfinite(values)) or np.any(self.plate_radii <= 0)
                or not 0 < 2 * self.wall < self.width or self.height <= self.wall
                or self.plate_height <= 0):
            raise ValueError("invalid finite tableware dimensions")
        self.fixtures = self._make()

    def _make(self):
        white = (0.91, 0.93, 0.90)
        base = trimesh.creation.cylinder(radius=1., height=self.wall, sections=64)
        base.apply_scale([*self.plate_radii, 1.])
        base.apply_translation([*self.plate_center, self.floor - self.wall / 2])
        plate = [base]
        # Individually convex sloping rim sectors preserve the empty dish interior.
        for i in range(48):
            angle = np.arange(i, i + 2) * 2 * np.pi / 48
            vertices = []
            for radius, z in ((.86, self.floor - self.wall),
                              (.86, self.floor), (1.0, self.floor + self.plate_height),
                              (1.0, self.floor + self.plate_height - self.wall)):
                vertices.extend(np.c_[self.plate_center + radius * self.plate_radii
                                      * np.c_[np.cos(angle), np.sin(angle)], [z, z]])
            plate.append(trimesh.convex.convex_hull(np.asarray(vertices)))
        fixtures = [Fixture("plate", trimesh.util.concatenate(plate), plate, white)]
        rails = []
        for i, (x, y) in enumerate(self.centers):
            w, t, h, z = self.width, self.wall, self.height, self.floor
            pieces = [_box([w, w, t], [x, y, z - t / 2])]
            for sign in (-1, 1):
                pieces.append(_box([t, w, h], [x + sign * (w - t) / 2, y, z + h / 2]))
                pieces.append(_box([w - 2*t, t, h], [x, y + sign * (w - t) / 2, z + h / 2]))
                rails.append(_box([w, .002, .002], [x, y + sign * w / 2, z + h]))
                rails.append(_box([.002, w, .002], [x + sign*w/2, y, z + h]))
            fixtures.append(Fixture(f"tomato_box_{i}", trimesh.util.concatenate(pieces), pieces, white))
        # Decorative gold rack; the white boxes provide the physical containment.
        xmin, xmax = self.centers[:, 0].min() - self.width/2, self.centers[:, 0].max() + self.width/2
        y = self.centers[:, 1].mean()
        for sign in (-1, 1):
            rails.append(_box([xmax-xmin + .02, .003, .003],
                              [(xmin+xmax)/2, y + sign*(self.width/2+.004), self.floor-.002]))
        fixtures.append(Fixture("gold_rack", trimesh.util.concatenate(rails), [], (.78, .56, .15)))
        desktop = self.cfg.get("tabletop", {})
        if desktop.get("enabled", False):
            center = np.asarray(desktop["center_xy"], dtype=float)
            size = np.asarray(desktop["size_xy_m"], dtype=float)
            thickness = float(desktop["thickness_m"])
            if (center.shape != (2,) or size.shape != (2,)
                    or not np.all(np.isfinite(np.r_[center, size, thickness]))
                    or np.any(size <= 0) or thickness <= 0):
                raise ValueError("tabletop requires finite XY center and positive dimensions")
            # The existing infinite ground plane supplies contact. A 0.2 mm
            # visual offset avoids coplanar ray/depth flicker without moving props.
            top = self.floor - self.wall + .0002
            slab = _box([*size, thickness], [*center, top-thickness/2])
            fixtures.append(Fixture("tabletop", slab, [], (.52, .30, .15)))
            # Code-native thin curved veneer strips: deterministic stylized
            # wood grain without a downloaded texture or additional colliders.
            shades = [[], [], []]
            ys = np.linspace(center[1]-size[1]/2, center[1]+size[1]/2, 64)
            for k, x in enumerate(np.linspace(center[0]-.48*size[0], center[0]+.48*size[0], 100)):
                curve = x + .0015*np.sin(ys*12 + k*.47) + .0005*np.sin(ys*39+k)
                width = .0003 + .00025*(1 + np.sin(k*1.7))
                vertices = np.concatenate((np.c_[curve-width/2, ys, np.full(64, top+.00002)],
                                           np.c_[curve+width/2, ys, np.full(64, top+.00002)]))
                faces = []
                for j in range(63):
                    faces.extend(((j, j+64, j+1), (j+1, j+64, j+65)))
                shades[k % 3].append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
            for i, color in enumerate(((.43, .23, .10), (.48, .26, .12), (.57, .34, .17))):
                fixtures.append(Fixture(f"tabletop_grain_{i}", trimesh.util.concatenate(shades[i]), [], color))
        return fixtures

    def apply_tabletop_gain(self, gain):
        """Tint render fixtures after exporting reusable nominal geometry assets."""
        gain = float(gain)
        if not np.isfinite(gain) or not 0 < gain <= 1.:
            raise ValueError("tabletop gain must be in (0, 1]")
        for fixture in self.fixtures:
            if fixture.name.startswith("tabletop"):
                fixture.color = tuple(channel * gain for channel in fixture.color)

    def inside(self, position, radius, index):
        p = np.asarray(position)
        half = self.width/2 - self.wall - radius
        return bool(half > 0 and np.all(np.abs(p[:2] - self.centers[index]) < half)
                    and self.floor + radius - .004 <= p[2] < self.floor + self.height)

    def export(self, root: Path):
        """Export reusable visual OBJ, convex collision OBJ and a fixed MJCF scene."""
        key = sha256(repr(self.cfg).encode()).hexdigest()[:12]
        directory = root / f"tableware_{key}"
        directory.mkdir(parents=True, exist_ok=True)
        mjcf = ET.Element("mujoco", model="tomato_tableware")
        asset, world = ET.SubElement(mjcf, "asset"), ET.SubElement(mjcf, "worldbody")
        paths = {}
        for fixture in self.fixtures:
            visual = directory / f"{fixture.name}.obj"
            if not visual.exists():
                fixture.mesh.export(visual)
            collisions = []
            for i, mesh in enumerate(fixture.colliders):
                name = f"{fixture.name}_collision_{i:02d}"
                path = directory / f"{name}.obj"
                if not path.exists():
                    mesh.export(path)
                collisions.append(path)
                ET.SubElement(asset, "mesh", name=name, file=path.name)
                ET.SubElement(world, "geom", type="mesh", mesh=name,
                              rgba=" ".join(map(str, (*fixture.color, 1.))))
            paths[fixture.name] = (visual, tuple(collisions))
            if not collisions:
                ET.SubElement(asset, "mesh", name=fixture.name, file=visual.name, inertia="shell")
                ET.SubElement(world, "geom", type="mesh", mesh=fixture.name,
                              contype="0", conaffinity="0",
                              rgba=" ".join(map(str, (*fixture.color, 1.))))
        combined = trimesh.util.concatenate([f.mesh for f in self.fixtures
                                             if f.name.startswith("tomato_box_") or f.name == "gold_rack"])
        if not (directory / "tomato_boxes.obj").exists():
            combined.export(directory / "tomato_boxes.obj")
        ET.ElementTree(mjcf).write(directory / "tableware.xml", encoding="unicode")
        self.asset_directory = directory
        return paths


class DepositMotion:
    """Ground-truth scripted transport; never moves the fruit body directly."""
    def __init__(self, tableware, cfg):
        self.tableware = tableware
        self.index = cfg.get("target_box", 1)
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index not in (0, 1, 2):
            raise ValueError("deposit.target_box must be 0, 1, or 2")
        self.travel_s = float(cfg.get("travel_s", 1.6))
        self.release_s = float(cfg.get("release_s", .4))
        self.drop_z = float(cfg.get("drop_center_z_m", .18))
        if (not np.all(np.isfinite([self.travel_s, self.release_s, self.drop_z]))
                or min(self.travel_s, self.release_s) <= 0
                or self.drop_z <= tableware.floor + tableware.height + .03):
            raise ValueError("deposit timings must be positive and release must clear the rim")
        self.start = None
        self.release_time = None
        self.phase = "deposit_idle"
        self.settle_s = 0.
        self.placed = False
        self.failed = False

    def command(self, time, tcp, fruit, radius, grasped):
        if self.start is None:
            self.start = time
            self.origin = tcp.copy()
            self.goal = tcp.copy()
            destination = np.r_[self.tableware.centers[self.index], max(self.drop_z, fruit[2])]
            self.goal[:3] += destination - fruit
        fraction = np.clip((time-self.start)/self.travel_s, 0., 1.)
        pose = self.origin.copy()
        pose[:3] += fraction*fraction*(3-2*fraction)*(self.goal[:3]-self.origin[:3])
        if self.failed:
            return tcp.copy(), float(self.release_time is None), "deposit_failed"
        if self.placed:
            return self.goal.copy(), 0., "deposit_complete"
        if self.release_time is None:
            self.phase = "deposit_transport"
            at_box = (np.all(np.abs(fruit[:2]-self.tableware.centers[self.index])
                             < self.tableware.width/2-self.tableware.wall-radius-.004)
                      and fruit[2] > self.tableware.floor+self.tableware.height+radius
                      and np.linalg.norm(tcp[:3]-self.goal[:3]) < .004)
            if fraction >= 1 and at_box and grasped:
                self.release_time = time
            elif time - self.start > self.travel_s + 1.:
                self.failed = True
                return tcp.copy(), 1., "deposit_failed"
        if self.release_time is not None:
            if time - self.release_time > 3.:
                self.failed = True
                return self.goal.copy(), 0., "deposit_failed"
            amount = np.clip((time-self.release_time)/self.release_s, 0., 1.)
            self.phase = "deposit_release" if amount < 1 else "deposit_settle"
            return self.goal.copy(), float(1-amount), self.phase
        return pose, 1., self.phase

    def observe(self, position, velocity, radius, dt):
        if self.failed:
            self.placed = False
            return
        if self.release_time is None:
            return
        settled = self.tableware.inside(position, radius, self.index) and np.linalg.norm(velocity) < .08
        self.settle_s = self.settle_s + dt if settled else 0.
        self.placed = self.settle_s >= .3
        if self.placed:
            self.phase = "deposit_complete"
