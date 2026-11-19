# Tomato Pick Sim contributor guide

This is the standalone tomato-harvesting checkout. Preserve the original
LycheeSim project; do not edit it as a side effect of work here. Keep new
features local to this project and do not reintroduce imports from other demos.

Read README.md and the relevant task module before changing behavior. Use
`configs/cherry_tomato_pick_sharpa.yaml` for the current harvesting/deposit
workflow. Preserve automatic recording, unique episode naming, randomized
scene generation, target conditioning and quality filtering unless asked.

Use Python 3.11 and `uv sync --locked`. Validate with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest`. For physics changes,
also run a bounded headless simulation and distinguish startup/recording checks
from a completed, quality-accepted harvest. Do not launch long collection or
training jobs or push remotely without explicit user authorization.

Preserve user edits and data. Do not recursively delete files/directories.
Do not overwrite episodes; use the recorder/batch output allocation paths.
Keep documentation in English and preserve provenance and third-party licenses.
