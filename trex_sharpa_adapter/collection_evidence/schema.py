"""Sharpa Wave joint ordering and limits."""

from litchi_hardware.core.types import JointSchema

_JOINT_NAMES: tuple[str, ...] = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)

_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-0.1745, 1.9199),
    (-0.3491, 0.3491),
    (-0.5236, 1.3963),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (0.0, 0.2618),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
)

SHARPA_WAVE_JOINT_SCHEMA = JointSchema(
    id="sharpa_wave_22_v1",
    names=_JOINT_NAMES,
    unit="rad",
    minimum=tuple(lower for lower, _ in _JOINT_LIMITS),
    maximum=tuple(upper for _, upper in _JOINT_LIMITS),
)

SHARPA_OPEN_POSITION: tuple[float, ...] = (0.0,) * SHARPA_WAVE_JOINT_SCHEMA.size
SHARPA_CLOSE_POSITION: tuple[float, ...] = tuple(
    max(0.0, lower + 0.7 * (upper - lower))
    for lower, upper in SHARPA_WAVE_JOINT_SCHEMA.limits
)
