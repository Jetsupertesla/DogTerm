"""Tests for parameter validation (hard checks + kinematic screen)."""

import math

import pytest

from roboevolve.controller import DEFAULT_CONTROLLER_PARAMS
from roboevolve.parameter_validator import (kinematic_feasibility, param_distance,
                                            validate_controller_params)


def test_valid_partial_proposal_merges_over_base():
    r = validate_controller_params({"frequency_hz": 1.2, "stride_scale": 0.8})
    assert r.ok, r.errors
    assert r.params["frequency_hz"] == 1.2
    assert set(r.params) == set(DEFAULT_CONTROLLER_PARAMS)


@pytest.mark.parametrize("proposal,fragment", [
    ({"frequncy_hz": 1.0}, "unsupported"),
    ({"torque_boost": 2.0}, "unsupported"),
    ({"frequency_hz": 9.0}, "outside"),
    ({"kp": -0.1}, "outside"),
    ({"kd": 5.0}, "outside"),          # unreasonable actuator gain
    ({"frequency_hz": float("nan")}, "NaN"),
    ({"step_height_m": float("inf")}, "NaN"),
    ({"stride_scale": "big"}, "not a number"),
    ({"stride_scale": True}, "not a number"),
    ({"phase_offsets": [0, 0.5, 0.5]}, "list of 4"),
    ({"phase_offsets": [0, 0.5, 0.5, 1.2]}, "[0, 1)"),
    ({"phase_offsets": [0, 0.5, math.nan, 0]}, "non-finite"),
    ({"phase_offsets": "trot"}, "list of 4"),
])
def test_invalid_proposals_rejected(proposal, fragment):
    r = validate_controller_params(proposal)
    assert not r.ok
    assert r.params is None
    assert any(fragment in e for e in r.errors), r.errors


def test_missing_required_parameters():
    r = validate_controller_params({"frequency_hz": 1.0}, require_all=True)
    assert not r.ok and any("missing" in e for e in r.errors)
    assert validate_controller_params(dict(DEFAULT_CONTROLLER_PARAMS), require_all=True).ok


def test_non_dict_rejected():
    assert not validate_controller_params(["frequency_hz", 1.0]).ok


def test_rejects_rather_than_clips():
    r = validate_controller_params({"frequency_hz": 4.01})
    assert not r.ok


def test_kinematic_feasibility():
    assert kinematic_feasibility(validate_controller_params({}).params).ok
    fast = validate_controller_params({"frequency_hz": 4.0, "stride_scale": 2.0,
                                       "duty_factor": 0.8, "step_height_m": 0.12}).params
    r = kinematic_feasibility(fast)
    assert not r.ok and "exceeds actuator limit" in r.errors[0]


def test_param_distance_wraps_phase():
    a = validate_controller_params({"phase_offsets": [0.0, 0.5, 0.5, 0.0]}).params
    b = validate_controller_params({"phase_offsets": [0.99, 0.5, 0.5, 0.0]}).params
    assert param_distance(a, a) == 0.0
    assert param_distance(a, b) < 0.01
