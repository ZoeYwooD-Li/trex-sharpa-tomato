"""Focused checkout CLI; no discovery/import of unrelated LycheeSim demos."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/cherry_tomato_pick_sharpa.yaml"


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    # Demos/assets belong to this checkout, never another installed LycheeSim.
    sys.path.insert(0, str(ROOT))
    if not args or args[0] in {"-h", "--help"}:
        print("Usage: tomato-sim {run|batch|quality|eval} [options]\n"
              "  run      One harvesting + deposit attempt; records HDF5 by default.\n"
              "  batch    Repeated independent attempts (e.g. --attempts 100).\n"
              "  quality  Inspect recorded episode/batch quality.\n"
              "  eval     One recorded OpenPI absolute-action policy rollout.\n"
              "Run commands from this project directory. Use <command> --help for options.")
        return 0
    command, args = args[0], args[1:]
    try:
        if command == "eval":
            from demos.cherry_tomato_pick.openpi_eval import main as evaluate
            return evaluate(args)
        if command == "run":
            # Keep the existing batch subprocess and familiar run command working.
            if args and args[0] in {"cherry-tomato-pick", "cherry_tomato_pick"}:
                args.pop(0)
            from demos.cherry_tomato_pick.scene import build
            from lycheesim.engine.runner import main as run
            run(build, demo_name="cherry_tomato_pick", argv=args,
                default_config=DEFAULT_CONFIG, prog="tomato-sim run")
            return 0
        if command == "batch":
            from demos.cherry_tomato_pick.batch import main as batch
            return batch(args)
        if command == "quality":
            from demos.cherry_tomato_pick.quality import main as quality
            return quality(args)
        raise ValueError(f"unknown command {command!r}; choose run, batch, quality or eval")
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(f"tomato-sim: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
