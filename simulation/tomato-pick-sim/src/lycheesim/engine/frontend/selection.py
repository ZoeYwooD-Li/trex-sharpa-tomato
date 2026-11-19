"""Select the supported viewer/headless/NPZ sinks without other demo backends."""
from . import HeadlessFrontend, RecorderFrontend, RerunFrontend


def select_frontends(cfg, args, demo_name, dt, every):
    render = cfg.get("render", {})
    if args.frontend is not None:
        names = [value.strip() for value in args.frontend.split(",") if value.strip()]
        if not names:
            raise ValueError("--frontend cannot be empty")
    else:
        names = [] if args.no_render else ["rerun"]
        if args.record is not None:
            names.append("recorder")
        if not names:
            names.append("headless")
    frontends = []
    for name in names:
        if name == "rerun":
            camera = render.get("camera", {}) or {}
            frontends.append(RerunFrontend(
                out_path=render.get("out_path"), fps=render.get("fps", 60),
                live=args.live, view_up=render.get("view_up", "y"),
                eye_position=camera.get("position"), look_target=camera.get("look_target"),
            ))
        elif name == "headless":
            frontends.append(HeadlessFrontend())
        elif name == "recorder":
            if args.record is None:
                raise ValueError("frontend 'recorder' requires --record <path.npz>")
            frontends.append(RecorderFrontend(args.record, meta={
                "dt": float(dt), "every": int(every), "fps": float(1.0 / (dt * every)),
                "demo": demo_name, "view_up": render.get("view_up", "y"),
            }))
        else:
            raise ValueError(f"unsupported frontend {name!r}; choose rerun, headless or recorder")
    return frontends
