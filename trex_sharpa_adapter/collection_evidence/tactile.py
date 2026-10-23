"""ROS-independent Sharpa fingertip tactile values and SDK decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from litchi_hardware.core._compat import StrEnum
from litchi_hardware.core.types import HandSide


class SharpaFinger(StrEnum):
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"


SHARPA_FINGERS = tuple(SharpaFinger)


def tactile_channel(side: HandSide, finger: SharpaFinger) -> int:
    """Return the SDK tactile channel for a physical finger and hand side.

    The channel index refers to the SDK's tactile stream, keyed by the
    *physical* finger. Right hands are laid out thumb..pinky over channels
    0..4. On the Wave left hand the fingertip channels are wired in the
    reverse order (field-verified on serial CD559038CD54 / Wave-L-01):
    physical thumb..pinky map to channels 9..5.
    """

    index = SHARPA_FINGERS.index(finger)
    if side is HandSide.LEFT:
        return 9 - index
    return index


@dataclass(frozen=True, slots=True)
class TactileImage:
    """Contiguous single-channel image bytes."""

    width: int
    height: int
    data: bytes
    encoding: str = "mono8"

    def __post_init__(self) -> None:
        data = bytes(self.data)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("tactile image dimensions must be positive")
        if self.encoding != "mono8":
            raise ValueError("only mono8 tactile images are currently supported")
        if len(data) != self.width * self.height:
            raise ValueError("tactile image byte count does not match its dimensions")
        object.__setattr__(self, "data", data)

    @property
    def step(self) -> int:
        return self.width


@dataclass(frozen=True, slots=True)
class SharpaTactileFrame:
    """One SDK tactile frame for one fingertip."""

    side: HandSide
    finger: SharpaFinger
    channel: int
    frame_index: int
    timestamp_s: float
    raw: TactileImage | None = None
    deformation: TactileImage | None = None
    wrench: tuple[float, ...] = ()
    contacts: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.channel != tactile_channel(self.side, self.finger):
            raise ValueError("tactile channel does not match hand side and finger")
        if self.frame_index < 0:
            raise ValueError("tactile frame index must not be negative")
        if not math.isfinite(self.timestamp_s):
            raise ValueError("tactile timestamp must be finite")
        wrench = tuple(float(value) for value in self.wrench)
        if wrench and (
            len(wrench) != 6 or not all(math.isfinite(value) for value in wrench)
        ):
            raise ValueError("tactile wrench must contain six finite values")
        contacts = tuple(
            tuple(float(value) for value in contact) for contact in self.contacts
        )
        if any(
            len(contact) != 3
            or not all(math.isfinite(value) for value in contact)
            for contact in contacts
        ):
            raise ValueError("tactile contacts must contain u, v, confidence")
        object.__setattr__(self, "wrench", wrench)
        object.__setattr__(self, "contacts", contacts)


def _nested_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(size) for size in shape)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return (len(value),)
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    child_shape = _nested_shape(value[0])
    if any(_nested_shape(child) != child_shape for child in value[1:]):
        raise ValueError("tactile array is ragged")
    return (len(value), *child_shape)


def _flatten(value: Any) -> list[Any]:
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        return list(value.reshape(-1).tolist())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return list(bytes(value))
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for child in value:
            result.extend(_flatten(child))
        return result
    return [value]


def _mono8_image(value: Any, expected_shape: tuple[int, int]) -> TactileImage | None:
    if value is None:
        return None
    shape = _nested_shape(value)
    while len(shape) > 2 and shape[0] == 1:
        shape = shape[1:]
    expected_size = math.prod(expected_shape)
    value_size = math.prod(shape) if shape else 0
    if len(shape) == 1 and value_size == expected_size:
        shape = expected_shape
    if shape != expected_shape:
        if len(shape) != 2 or math.prod(shape) != value_size:
            raise ValueError(
                f"unexpected tactile image shape {shape}; expected {expected_shape}"
            )
    if hasattr(value, "astype") and hasattr(value, "tobytes"):
        array = value.astype("uint8", copy=False)
        data = array.tobytes(order="C")
    else:
        flat = _flatten(value)
        data = bytes(max(0, min(255, int(item))) for item in flat)
    return TactileImage(width=shape[1], height=shape[0], data=data)


def _float_values(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    return tuple(float(item) for item in _flatten(value))


def _contacts(value: Any) -> tuple[tuple[float, float, float], ...]:
    values = _float_values(value)
    if not values:
        return ()
    if len(values) % 3:
        raise ValueError("tactile CONTACT_POINT data is not divisible into triples")
    return tuple(
        (values[index], values[index + 1], values[index + 2])
        for index in range(0, len(values), 3)
    )


def decode_tactile_frame(
    payload: Any, side: HandSide, finger: SharpaFinger
) -> SharpaTactileFrame:
    """Decode the dictionary returned by Sharpa's Python SDK."""

    if not isinstance(payload, dict):
        raise TypeError("Sharpa tactile payload must be a dictionary")
    content = payload.get("content") or {}
    if not isinstance(content, dict):
        raise TypeError("Sharpa tactile content must be a dictionary")
    return SharpaTactileFrame(
        side=side,
        finger=finger,
        channel=int(payload.get("channel", tactile_channel(side, finger))),
        frame_index=int(payload.get("frame_id", 0)),
        timestamp_s=float(payload.get("ts", 0.0)),
        raw=_mono8_image(content.get("RAW"), (240, 320)),
        deformation=_mono8_image(content.get("DEFORM"), (240, 240)),
        wrench=_float_values(content.get("F6")),
        contacts=_contacts(content.get("CONTACT_POINT")),
    )
