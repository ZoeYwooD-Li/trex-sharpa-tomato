"""Planar scripted recenter limits; not a physical grasp-success test."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from demos.cherry_tomato_pick.scene import _bounded_recenter_xy, build


def test_recenter_step_is_rate_limited_and_does_not_mutate_inputs():
    current = np.array([.3, -.1])
    anchor = current.copy()
    desired = current + [.03, .04]
    result = _bounded_recenter_xy(current, desired, anchor, max_step=.000025, max_offset=.02)
    assert np.linalg.norm(result-current) == pytest.approx(.000025)
    np.testing.assert_array_equal(current, anchor)


def test_recenter_anchor_caps_total_motion_even_when_target_keeps_moving():
    anchor = np.array([.3, -.1])
    current = anchor.copy()
    for _ in range(2000):
        current = _bounded_recenter_xy(current, current+[.01, .01], anchor,
                                       max_step=.000025, max_offset=.02)
        assert np.linalg.norm(current-anchor) <= .02000000001
    assert np.linalg.norm(current-anchor) == pytest.approx(.02)


def test_recenter_does_not_overshoot_a_nearby_target():
    np.testing.assert_allclose(_bounded_recenter_xy([0., 0.], [.001, 0.], [0., 0.],
                                                   max_step=.01, max_offset=.02), [.001, 0.])


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_invalid_recenter_limits_fail_before_runtime_allocation(value):
    cfg = yaml.safe_load(Path("configs/cherry_tomato_pick_deposit.yaml").read_text())
    cfg = deepcopy(cfg)
    cfg["script"].update(support_recenter_while_closing=True, support_recenter_speed_m_s=value)
    with pytest.raises(ValueError, match="recenter"):
        build(cfg)


def test_invalid_xy_is_rejected():
    with pytest.raises(ValueError, match="recenter"):
        _bounded_recenter_xy([np.nan, 0.], [0., 0.], [0., 0.], max_step=.01, max_offset=.02)
