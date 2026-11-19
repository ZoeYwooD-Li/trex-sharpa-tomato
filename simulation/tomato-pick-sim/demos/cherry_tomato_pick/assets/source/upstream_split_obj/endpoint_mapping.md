# Cherry tomato endpoint mapping

The stable endpoint definition follows the source asset's longitudinal local
Z axis. Screen-space left/right changes with the selected camera.

| Selector | Original group | Local-Z center | Paired tomato at that end |
|---|---:|---:|---:|
| `end_min` | `tomato_07` | -10.992692 | `tomato_02` |
| `end_max` | `tomato_10` | 10.275068 | `tomato_09` |

For the current assembled transform (`rotateXYZ = (90, 0, 0)`), local Z maps
to world -Y. The prototype accepts either endpoint name or the original group
name. Camera-facing aliases `leftmost` and `rightmost` should only be assigned
at the task layer, where the active camera is known.
