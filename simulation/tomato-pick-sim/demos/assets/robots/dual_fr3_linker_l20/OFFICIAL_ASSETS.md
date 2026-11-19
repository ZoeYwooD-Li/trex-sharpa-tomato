# Official FR3 and Sharpa assets

The FR3 arm meshes already present under `meshes/fr3v2` were compared byte for
byte with the current Franka Robotics `franka_description` repository and are
unchanged.

- Source: <https://github.com/frankarobotics/franka_description>
- Verified revision: `02afaae282d4a8e10d7d2f781b23b3515c303ce5`
- Robot family: `meshes/robots/fr3v2`
- License: Apache-2.0; see `LICENSE.franka.txt`

Only meshes loaded by the active demos are retained locally: the FR3 visual and
collision links, the dual-robot base, and the LinkerHand collision meshes. The
arm mount frames in `configs/power_screwdriver.yaml` are copied from the
matching official `kinematics.yaml`.

Sharpa Wave assets are documented separately in
`../sharpa_wave_official/PROVENANCE.md` and match the current official revision
`6eea427eb24189519f32b9f21674cd534d3f973c`.

Neither official model repository ships bitmap albedo images for these DAE/STL
parts. The demo therefore uses only the DAE/MJCF source colors and does not add
generated robot textures.
