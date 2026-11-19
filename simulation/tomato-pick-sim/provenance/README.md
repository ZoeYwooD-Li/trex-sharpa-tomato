# Extraction provenance and third-party assets

This project copies the current working-tree tomato implementation from
LycheeSim, upstream <https://github.com/louhz/lycheesim>, branch `camera`, base
commit `2d75161980a4f2ab6bf9e30dd746aef7046a3d6d`. Uncommitted changes are included.
The machine-readable [source snapshot](source_snapshot.json) lists their status
and SHA-256 hashes of every copied source file before standalone edits.

Standalone edits are limited to dependency/CLI packaging, narrowing runtime
imports/frontends, extracting inherited kinematic helpers, and documenting the
new checkout. FK/mesh helpers previously imported from `lemon_cut/dual_wuji.py`
and `lemon_slice_pick/kinematics.py` and `motion.py` are now task-local in
`demos/cherry_tomato_pick/kinematics.py`; the unused Wuji v1 skin/IK methods and
lemon-specific logic are not included. The Beta2 subclass still supplies its
own mounting, IK and fingertip geometry.

## Attribution and redistribution

- Wuji Hand 2 Beta2 assets are from
  <https://github.com/wuji-technology/wuji-description>, revision
  `4c1073d0a3ad1daaf6546d219db751d8448d3888`. The original `LICENSE.txt` and
  `PROVENANCE.md` are included in `demos/assets/robots/wuji_hand2_beta2_official/`.
- The carrier's Franka license and adapter provenance are retained under
  `demos/assets/robots/dual_fr3_linker_l20/`. The carrier XML also references
  Linker-hand meshes, so those referenced source meshes remain even though
  the active tomato hands are the separately mounted official Wuji2 models.
- Tomato runtime metadata identifies source repository
  `git@github.com:yxsdev/cherry_tomato_sim.git`, revision
  `43610761c4f667dc24ac6b5f7d66b71ebf72c1a1`, source transforms, and mesh hashes.
  The source checkout did not contain the `ASSET_PROVENANCE.md` mentioned by its
  task README, nor a root LICENSE file. No missing license or redistribution
  permission is invented here. Confirm permissions with the source/asset owner
  before publishing this project or sending its assets to others.

This local extraction does not relicense upstream code or third-party assets.
