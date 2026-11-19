"""Newton explicit-contact-manifold sensor adapter.

This module deliberately wraps the public :mod:`newton.sensors` API instead
of copying Newton's private ``_src`` implementation into LycheeSim.
"""

from __future__ import annotations

import inspect
from typing import Any

from ..base import SensorSpec
from .base import ContactSensorBase


class NewtonContactSensor(ContactSensorBase):
    """Measure mesh--mesh contact forces through Newton ``SensorContact``.

    The public attributes mirror Newton's sensor and remain Warp arrays on the
    model device.  In particular, ``total_force`` and ``force_matrix`` contain
    world-frame forces in newtons.

    Construct this sensor after finalizing the model but *before* constructing
    the ``Contacts`` object, so Newton can request its ``force`` attribute.
    """

    def __init__(
        self,
        model: Any,
        *,
        sensing_bodies: str | list[str] | list[int] | None = None,
        sensing_shapes: str | list[str] | list[int] | None = None,
        counterpart_bodies: str | list[str] | list[int] | None = None,
        counterpart_shapes: str | list[str] | list[int] | None = None,
        measure_total: bool = True,
        verbose: bool | None = None,
        _sensor_cls: type | None = None,
    ) -> None:
        if _sensor_cls is None:
            from newton.sensors import SensorContact

            _sensor_cls = SensorContact

        parameters = inspect.signature(_sensor_cls).parameters
        kwargs: dict[str, Any] = {
            "counterpart_bodies": counterpart_bodies,
            "counterpart_shapes": counterpart_shapes,
            "measure_total": measure_total,
            "verbose": verbose,
        }
        if "sensing_bodies" in parameters:
            kwargs["sensing_bodies"] = sensing_bodies
            kwargs["sensing_shapes"] = sensing_shapes
        else:
            # Newton <=1.2 used the ``sensing_obj_*`` spelling.
            kwargs["sensing_obj_bodies"] = sensing_bodies
            kwargs["sensing_obj_shapes"] = sensing_shapes

        self.model = model
        self.native_sensor = _sensor_cls(
            model, **{name: value for name, value in kwargs.items() if value is not None}
        )

    @property
    def spec(self) -> SensorSpec:
        return SensorSpec(
            sensor_type="contact",
            adapter="newton_explicit_contact_force",
            input_fields=(
                "state.body_q",
                "contacts contact manifold",
                "contacts.force",
            ),
            output_fields=(
                "total_force_N",
                "force_matrix_N",
                "friction_force_N",
                "contact_position",
            ),
            differentiable=False,
        )

    @property
    def device(self):
        return self.native_sensor.device

    @property
    def total_force(self):
        return self.native_sensor.total_force

    @property
    def force_matrix(self):
        return self.native_sensor.force_matrix

    @property
    def total_force_friction(self):
        return self.native_sensor.total_force_friction

    @property
    def force_matrix_friction(self):
        return self.native_sensor.force_matrix_friction

    @property
    def position_matrix(self):
        """Force-weighted contact positions, when provided by Newton."""
        return getattr(self.native_sensor, "position_matrix", None)

    @property
    def sensing_transforms(self):
        if hasattr(self.native_sensor, "sensing_transforms"):
            return self.native_sensor.sensing_transforms
        return self.native_sensor.sensing_obj_transforms

    @property
    def sensing_indices(self) -> list[int]:
        if hasattr(self.native_sensor, "sensing_indices"):
            return list(self.native_sensor.sensing_indices)
        return list(self.native_sensor.sensing_obj_idx)

    @property
    def sensing_type(self) -> str:
        if hasattr(self.native_sensor, "sensing_type"):
            return str(self.native_sensor.sensing_type)
        return str(self.native_sensor.sensing_obj_type)

    @property
    def counterpart_indices(self) -> list[list[int]]:
        return self.native_sensor.counterpart_indices

    @property
    def counterpart_type(self) -> str | None:
        return self.native_sensor.counterpart_type

    def update(self, state: Any, contacts: Any, *, solver: Any | None = None) -> None:
        """Update contact forces on the model device.

        When ``solver`` is supplied, its ``update_contacts`` hook is called
        first.  This is required by force-reporting solvers such as MuJoCo and
        XPBD after their step.
        """
        if solver is not None:
            solver.update_contacts(contacts, state)
        self.native_sensor.update(state, contacts)
