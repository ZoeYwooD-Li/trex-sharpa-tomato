"""One bounded, recorded OpenPI rollout through the existing simulator runner."""
import argparse
import json

from .openpi_policy import OpenPIClient, PolicyController


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tomato-sim eval", description=__doc__, epilog=(
        "Remaining options are passed to tomato-sim run, e.g. --frontend rerun --live "
        "or --duration 28 --set recording.seed=42."))
    parser.add_argument("--policy-url", default="ws://127.0.0.1:8000")
    parser.add_argument("--request-timeout", type=float, default=180.)
    parser.add_argument("--execute-steps", type=int, default=4)
    parser.add_argument("--contact-stop-force-n", type=float, default=2000.,
                        help="Simulation-only debug stop threshold in N (default: 2000); does not relax quality acceptance")
    parser.add_argument("--disable-contact-stop", action="store_true",
                        help="Simulation-only: continue through finite contact-force overloads; keep other stops and quality checks")
    parser.add_argument("--grasp-before-lift", action="store_true",
                        help="Hybrid control: prevent right-hand lifting until the existing grasp confirmation passes")
    parser.add_argument("--grasp-wait-timeout", type=float, default=8.,
                        help="Maximum simulation seconds waiting for right grasp confirmation (default: 8)")
    parser.add_argument("--checkpoint-label", default="unspecified",
                        help="User-supplied provenance label, not server-verified identity")
    args, remaining = parser.parse_known_args(argv)
    from lycheesim.cli import DEFAULT_CONFIG
    from lycheesim.engine.runner import main as run
    from .scene import build

    # Validate before opening a connection; the transport imports are optional.
    controller = PolicyController(None, args.execute_steps, args.checkpoint_label, args.contact_stop_force_n,
                                  disable_contact_stop=args.disable_contact_stop,
                                  grasp_before_lift=args.grasp_before_lift,
                                  grasp_wait_timeout_s=args.grasp_wait_timeout)
    if args.grasp_before_lift:
        print(f"Hybrid grasp-before-lift mode enabled; wait timeout {args.grasp_wait_timeout:g}s simulation time.")
    if args.disable_contact_stop:
        print("WARNING: contact-force early stopping DISABLED for simulation debugging; "
              "nonfinite forces, other stops and dataset quality checks remain enforced.")
    client = OpenPIClient(args.policy_url, args.request_timeout)
    controller.client = client

    def build_policy(cfg, config_path):
        setup = build(cfg, config_path, policy=controller)
        recorder = setup._episode_recorder
        step, verify, close = setup.step, setup.verify, setup.close

        def step_policy(time_s, inputs):
            try:
                step(time_s, inputs)
            except Exception as exc:
                controller.error = f"simulation step failed: {exc}"
                raise

        def verify_policy():
            try:
                if controller.error:
                    raise RuntimeError(controller.error)
                verify()
                controller.verified = True
            except Exception as exc:
                controller.error = str(exc)
                raise

        def close_policy():
            saved = False
            try:
                recorder.metadata["policy_evaluation"] = controller.summary()
                close()
                saved = recorder.output_path.exists()
            except Exception as exc:
                controller.error = f"recording finalization failed: {exc}"
                raise
            finally:
                report = controller.summary()
                report.update(episode=str(recorder.output_path), seed=recorder.seed,
                              recording_saved=saved, quality=recorder.quality_report,
                              accepted=bool(saved and controller.verified and recorder.quality_report
                                            and recorder.quality_report.get("accepted")))
                controller.accepted = report["accepted"]
                path = recorder.output_path.with_suffix(".evaluation.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x") as stream:
                    json.dump(report, stream, indent=2)
                print(f"OpenPI evaluation report: {path}")

        setup.step, setup.verify, setup.close = step_policy, verify_policy, close_policy
        return setup

    defaults = ["--frontend", "headless", "--set",
                "recording.output_path=outputs/openpi_eval/episode_000001.h5"]
    try:
        run(build_policy, demo_name="cherry_tomato_pick", argv=defaults + remaining,
            default_config=DEFAULT_CONFIG, prog="tomato-sim eval")
    finally:
        client.close()
    return 0 if getattr(controller, "accepted", False) else 1
