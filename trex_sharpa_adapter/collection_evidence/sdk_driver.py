"""Adapter for the Sharpa Wave native Python SDK."""

from __future__ import annotations

import importlib
import math
import sys
import time
from typing import Any

from litchi_hardware.core.errors import (
    CommandRejectedError,
    ConnectionError,
    DependencyUnavailableError,
)
from litchi_hardware.core.lifecycle import ConnectionState
from litchi_hardware.core.types import HandSide, JointState

from .base import SharpaHandBase
from .config import SharpaConfig
from .schema import SHARPA_WAVE_JOINT_SCHEMA
from .tactile import (
    SharpaFinger,
    SharpaTactileFrame,
    decode_tactile_frame,
    tactile_channel,
)
from .vendor import load_sharpa_sdk

_COEFFICIENT_ABS_TOLERANCE = 1e-4
_ENABLE_POLL_INTERVAL_S = 0.05
_ENABLE_SETTLE_TIMEOUT_S = 1.0


class SharpaSdkHand(SharpaHandBase):
    def __init__(self, config: SharpaConfig, sdk_module: Any = None) -> None:
        super().__init__(config)
        self._sdk = sdk_module
        self._manager: Any = None
        self._wave: Any = None
        self._tactile_available = False

    @property
    def transport_name(self) -> str:
        return "sharpa-python-sdk"

    def _load_sdk(self) -> Any:
        if self._sdk is not None:
            return self._sdk
        if self.config.sdk_python_path and self.config.sdk_python_path not in sys.path:
            sys.path.insert(0, self.config.sdk_python_path)
        try:
            self._sdk = (
                importlib.import_module("sharpa")
                if self.config.sdk_python_path
                else load_sharpa_sdk()
            )
        except (ImportError, OSError) as exc:
            raise DependencyUnavailableError(
                "Sharpa SDK is unavailable; install the vendor deb or set sdk_python_path"
            ) from exc
        return self._sdk

    @staticmethod
    def _check_error(error: Any, operation: str) -> None:
        if int(error.code) != 0:
            message = getattr(error, "message", f"code {error.code}")
            raise CommandRejectedError(f"Sharpa {operation} failed: {message}")

    def _choose_device(self, devices: list[Any], sdk: Any) -> Any:
        hand_devices = [
            device
            for device in devices
            if not hasattr(device, "device_type") or device.device_type == sdk.DeviceType.HAND
        ]
        if self.config.serial:
            hand_devices = [device for device in hand_devices if device.sn == self.config.serial]
        if self.config.side is not None:
            expected = (
                sdk.HandSide.LEFT if self.config.side is HandSide.LEFT else sdk.HandSide.RIGHT
            )
            hand_devices = [
                device
                for device in hand_devices
                if getattr(device, "hand_side", expected) == expected
            ]
        if not hand_devices:
            raise ConnectionError("no matching Sharpa Wave hand discovered")
        return hand_devices[0]

    def _read_setting(self, getter_name: str) -> Any:
        error, value = getattr(self._wave, getter_name)()
        self._check_error(error, getter_name)
        return value

    def _ensure_coefficient(self, name: str, observed: float, desired: float) -> None:
        if math.isclose(
            float(observed),
            desired,
            rel_tol=0.0,
            abs_tol=_COEFFICIENT_ABS_TOLERANCE,
        ):
            return
        setter_name = f"set_{name}_coeff"
        getter_name = f"get_{name}_coeff"
        self._check_error(getattr(self._wave, setter_name)(desired), setter_name)
        verified = float(self._read_setting(getter_name))
        if not math.isclose(
            verified,
            desired,
            rel_tol=0.0,
            abs_tol=_COEFFICIENT_ABS_TOLERANCE,
        ):
            raise CommandRejectedError(
                f"Sharpa {setter_name} verification failed: expected {desired}, got {verified}"
            )

    def _ensure_enum_setting(self, name: str, observed: Any, desired: Any) -> None:
        if observed == desired:
            return
        setter_name = f"set_control_{name}"
        getter_name = f"get_control_{name}"
        self._check_error(getattr(self._wave, setter_name)(desired), setter_name)
        verified = self._read_setting(getter_name)
        if verified != desired:
            raise CommandRejectedError(
                f"Sharpa {setter_name} verification failed: expected {desired!r}, got {verified!r}"
            )

    def _ensure_enabled(self) -> None:
        if bool(self._read_setting("get_enable_state")):
            return
        self._check_error(self._wave.set_enable_state(True), "set_enable_state")
        deadline = time.monotonic() + _ENABLE_SETTLE_TIMEOUT_S
        while time.monotonic() < deadline:
            time.sleep(_ENABLE_POLL_INTERVAL_S)
            if bool(self._read_setting("get_enable_state")):
                return
        raise CommandRejectedError(
            "Sharpa set_enable_state verification failed: expected True, got False"
        )

    def connect(self) -> None:
        if self.is_connected:
            return
        if self.connection_state is ConnectionState.FAILED:
            self._lifecycle.reset()
        self._lifecycle.begin_connect()
        try:
            sdk = self._load_sdk()
            self._manager = sdk.SharpaWaveManager.get_instance()
            deadline = time.monotonic() + self.config.discovery_timeout_s
            devices: list[Any] = []
            device = None
            while time.monotonic() < deadline:
                devices = list(self._manager.get_all_devices())
                try:
                    device = self._choose_device(devices, sdk)
                except ConnectionError:
                    time.sleep(0.1)
                else:
                    break
            if device is None:
                raise ConnectionError("no matching Sharpa Wave hand discovered")
            self._serial = str(device.sn)
            if hasattr(device, "hand_side"):
                self._side = (
                    HandSide.LEFT if device.hand_side == sdk.HandSide.LEFT else HandSide.RIGHT
                )
            supports_tactile = bool(
                hasattr(device, "has_fingertip_tactile") and device.has_fingertip_tactile()
            )
            initialize_tactile = self.config.enable_tactile and supports_tactile
            self._wave = self._manager.connect(device.sn, not initialize_tactile)
            self._tactile_available = initialize_tactile

            # These settings persist on firmware 3.0.6, which rejects some
            # redundant writes. Read all values first, write only mismatches,
            # and verify every write before starting the receive lifecycle.
            speed_coefficient = float(self._read_setting("get_speed_coeff"))
            current_coefficient = float(self._read_setting("get_current_coeff"))
            control_source = self._read_setting("get_control_source")
            control_mode = self._read_setting("get_control_mode")
            self._ensure_coefficient("speed", speed_coefficient, self.config.speed_coefficient)
            self._ensure_coefficient(
                "current", current_coefficient, self.config.current_coefficient
            )
            self._ensure_enum_setting("source", control_source, sdk.ControlSource.SDK)
            self._ensure_enum_setting("mode", control_mode, sdk.ControlMode.POSITION)
            self._ensure_enabled()
            if not self._wave.start():
                raise ConnectionError("Sharpa start() returned false")
            self._lifecycle.connected()
        except Exception as exc:
            self._cleanup()
            self._lifecycle.failed()
            if isinstance(exc, (ConnectionError, DependencyUnavailableError, CommandRejectedError)):
                raise
            raise ConnectionError(f"failed to connect Sharpa hand: {exc}") from exc

    def _cleanup(self) -> None:
        if self._wave is not None:
            try:
                self._wave.stop()
                if hasattr(self._wave, "destroy"):
                    self._wave.destroy()
            except Exception:
                pass
            self._wave = None
            self._tactile_available = False
        if self._manager is not None:
            try:
                if self._serial:
                    self._manager.disconnect(self._serial)
            except Exception:
                pass
            self._manager = None

    def disconnect(self) -> None:
        if self.connection_state is ConnectionState.DISCONNECTED:
            return
        self._cleanup()
        self._lifecycle.disconnected()

    def _write_joint_position(self, positions: tuple[float, ...]) -> None:
        if self._wave is None:
            raise ConnectionError("Sharpa SDK handle is unavailable")
        error = self._wave.set_joint_position(list(positions), self.config.interpolate)
        self._check_error(error, "set_joint_position")

    def read_joint_state(self) -> JointState:
        if not self.is_connected or self._wave is None:
            raise ConnectionError("Sharpa hand is not connected")
        state = self._wave.get_states()
        return JointState(
            SHARPA_WAVE_JOINT_SCHEMA.names,
            tuple(float(value) for value in state.angles),
            tuple(float(value) for value in state.velocities),
            tuple(float(value) for value in state.torques),
            timestamp_s=time.time(),
            schema=SHARPA_WAVE_JOINT_SCHEMA,
        )

    @property
    def tactile_available(self) -> bool:
        return self.is_connected and self._tactile_available

    def read_tactile(
        self, finger: SharpaFinger | str, timeout_s: float | None = 0.0
    ) -> SharpaTactileFrame | None:
        if not self.is_connected or self._wave is None:
            raise ConnectionError("Sharpa hand is not connected")
        if not self._tactile_available or self._side is None:
            raise ConnectionError("Sharpa fingertip tactile is unavailable")
        resolved_finger = SharpaFinger(finger)
        timeout = -1.0 if timeout_s is None else float(timeout_s)
        payload = self._wave.fetch_tactile_frame(
            tactile_channel(self._side, resolved_finger), timeout
        )
        if payload is None:
            return None
        return decode_tactile_frame(payload, self._side, resolved_finger)

    def stop(self) -> None:
        if self._wave is None:
            return
        if hasattr(self._wave, "set_enable_state"):
            self._check_error(self._wave.set_enable_state(False), "stop")
