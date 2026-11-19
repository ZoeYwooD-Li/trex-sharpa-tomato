# Sharpa Wave official assets

This directory contains unmodified Sharpa Robotics Wave model and tactile
sensor assets used by the `power_screwdriver` demo.

- Hand model source: <https://github.com/sharpa-robotics/sharpa-urdf-usd-xml>
- Hand model revision: `6eea427eb24189519f32b9f21674cd534d3f973c`
- Tactile asset source: <https://github.com/sharpa-robotics/sharpa-tactile-sensor-assets>
- Tactile asset revision: `865530a98a0ca0e69d177f2121833f8bb3ed94de`
- License: Apache License 2.0; see `LICENSE.txt` and `tactile/LICENSE`.

The official Wave MJCF assigns semantic RGBA materials to the housing and
elastomer meshes. It does not ship bitmap albedo maps for these STL files, so
the demo preserves those official material colors instead of inventing a
texture. The five `tactileSensor_*.obj` meshes are the contact surfaces; the
point/normal maps are retained for future spatial taxel reconstruction.
