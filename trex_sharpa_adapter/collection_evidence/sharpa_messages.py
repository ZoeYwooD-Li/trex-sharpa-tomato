"""Convert ROS-independent Sharpa tactile values into standard ROS messages."""

from __future__ import annotations

from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image

from litchi_hardware.core.types import HandSide
from litchi_hardware.hardware.sharpa import (
    SharpaFinger,
    SharpaTactileFrame,
    TactileImage,
)


def tactile_topic_prefix(side: HandSide, finger: SharpaFinger) -> str:
    """Return the ROS topic prefix for one physical fingertip."""

    return f"/sharpa/{side.value}/tactile/{finger.value}"


def tactile_frame_id(frame: SharpaTactileFrame) -> str:
    return f"{frame.side.value}_{frame.finger.value}_fingertip"


def validate_tactile_route(
    frame: SharpaTactileFrame,
    *,
    expected_side: HandSide,
    expected_finger: SharpaFinger,
) -> str:
    """Validate a frame before publishing it on a physical-finger topic.

    SDK channel selection stays in the HAL. This ROS-side guard only ensures
    that the returned frame's physical identity matches the polling route, so
    a backend regression cannot publish a valid frame under another finger's
    topic. The returned frame ID is safe to use for every message on the route.
    """

    if frame.side is not expected_side or frame.finger is not expected_finger:
        raise ValueError(
            "Sharpa tactile frame identity mismatch: expected "
            f"{expected_side.value}/{expected_finger.value}, received "
            f"{frame.side.value}/{frame.finger.value}"
        )
    return tactile_frame_id(frame)


def tactile_image_message(
    image: TactileImage, *, frame_id: str, stamp: RosTime
) -> Image:
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.height = image.height
    message.width = image.width
    message.encoding = image.encoding
    message.is_bigendian = False
    message.step = image.step
    message.data = image.data
    return message


def tactile_wrench_message(
    frame: SharpaTactileFrame, *, stamp: RosTime
) -> WrenchStamped:
    if len(frame.wrench) != 6:
        raise ValueError("Sharpa tactile frame has no six-axis wrench")
    message = WrenchStamped()
    message.header.stamp = stamp
    message.header.frame_id = tactile_frame_id(frame)
    (
        message.wrench.force.x,
        message.wrench.force.y,
        message.wrench.force.z,
        message.wrench.torque.x,
        message.wrench.torque.y,
        message.wrench.torque.z,
    ) = frame.wrench
    return message
