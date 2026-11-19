"""Discovery and lazy loading for demos in an editable LycheeSim checkout."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path


def repository_root() -> Path:
    """Use this checkout only, ignoring ambient upstream-project overrides."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "configs/cherry_tomato_pick_deposit.yaml").is_file():
        raise ValueError("standalone tomato checkout is incomplete; copy configs and demos with src")
    return root


@dataclass(frozen=True, slots=True)
class DemoDescriptor:
    """A lazily importable demo and its repository-owned default config."""

    module_name: str
    root: Path

    @property
    def name(self) -> str:
        return self.module_name.replace("_", "-")

    @property
    def module(self) -> str:
        return f"demos.{self.module_name}.scene"

    @property
    def run_path(self) -> Path:
        return self.root / "demos" / self.module_name / "run.py"

    @property
    def scene_path(self) -> Path:
        return self.root / "demos" / self.module_name / "scene.py"

    @property
    def config_path(self) -> Path:
        return self.root / "configs" / f"{self.module_name}.yaml"

    @property
    def problems(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.run_path.is_file():
            missing.append("run.py")
        if not self.scene_path.is_file():
            missing.append("scene.py")
        if not self.config_path.is_file():
            missing.append("config")
        return tuple(missing)

    @property
    def available(self) -> bool:
        return not self.problems

    def load_builder(self):
        """Import and return the demo's public ``build`` callable."""

        if self.problems:
            missing = ", ".join(self.problems)
            raise ValueError(f"demo {self.name!r} is incomplete: missing {missing}")
        module = import_module(self.module)
        build = getattr(module, "build", None)
        if not callable(build):
            raise ValueError(f"demo {self.name!r} does not expose callable build(cfg)")
        return build


def list_demos() -> tuple[DemoDescriptor, ...]:
    """Expose only the tomato task; historical YAML variants are not demos."""
    return (DemoDescriptor("cherry_tomato_pick", repository_root()),)


def get_demo(name: str) -> DemoDescriptor:
    """Resolve a CLI demo name, accepting both kebab-case and snake_case."""

    normalized = name.strip().lower().replace("-", "_")
    for demo in list_demos():
        if demo.module_name == normalized:
            return demo
    choices = ", ".join(demo.name for demo in list_demos())
    raise ValueError(f"unknown demo {name!r}; choose one of: {choices}")


__all__ = [
    "DemoDescriptor",
    "get_demo",
    "list_demos",
    "repository_root",
]
