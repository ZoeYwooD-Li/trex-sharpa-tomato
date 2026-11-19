"""Configuration dataclasses for the optional RSL-RL backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..managers.base import config_to_dict, strip_none


@dataclass
class RslRlModelCfg:
    hidden_dims: tuple[int, ...] = (128, 128, 128)
    activation: str = "elu"
    obs_normalization: bool = False
    cnn_cfg: dict[str, Any] | None = None
    distribution_cfg: dict[str, Any] | None = None
    rnn_type: str | None = None
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    class_name: str = "MLPModel"


@dataclass
class ZeroInitModelCfg(RslRlModelCfg):
    """Model cfg for :class:`.models.ZeroInitMLPModel` (zero-init action mean)."""

    zero_init_last_layer: bool = True
    class_name: str = "lycheesim.rl.rsl_rl.models:ZeroInitMLPModel"


@dataclass
class RslRlPpoAlgorithmCfg:
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: Literal["adaptive", "fixed"] = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.005
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    normalize_advantage_per_mini_batch: bool = False
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    share_cnn_encoders: bool = False
    class_name: str = "PPO"


@dataclass
class RslRlBaseRunnerCfg:
    seed: int = 42
    num_steps_per_env: int = 24
    max_iterations: int = 300
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)}
    )
    save_interval: int = 50
    experiment_name: str = "lycheesim"
    run_name: str = ""
    logger: Literal["wandb", "tensorboard"] = "tensorboard"
    wandb_project: str = "lycheesim"
    wandb_tags: tuple[str, ...] = ()
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    clip_actions: float | None = None
    upload_model: bool = False

    def to_dict(self, *, drop_none: bool = True) -> dict[str, Any]:
        value = config_to_dict(self)
        assert isinstance(value, dict)
        return strip_none(value) if drop_none else value


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "OnPolicyRunner"
    actor: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            }
        )
    )
    critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
