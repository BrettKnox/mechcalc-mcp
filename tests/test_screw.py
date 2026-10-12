"""The claim the whole arm rests on, checked instead of assumed.

"The boom holds position with the power off" is the reason this machine uses leadscrews rather than
servos. Nothing verified it. `engines/linear.py` decided self-locking with
`holding = force if efficiency <= 0.5`, keyed on an efficiency the designer typed in -- so the
conclusion came from the assumption, and the assumption had been chosen because the conclusion was
wanted. The lead angle, which is what actually decides it, was never computed anywhere.

Computed, the two candidate leads fall on opposite sides of the line, and the design had picked the
wrong one.
"""
from __future__ import annotations

import math

import pytest

from mechcalc_mcp import screw

GREASED = "steel_brass_greased"


# ── the finding ──────────────────────────────────────────────────────────────────────────────────

def test_the_chosen_four_mm_lead_does_not_self_lock():
    r = screw.analyse("T8", 4.0, friction=GREASED)
    assert r["self_locking"] is False
    assert r["lead_angle_deg"] == pytest.approx(10.31, abs=0.05)
    assert r["mu_needed_to_lock"] == pytest.approx(0.176, abs=0.002)
    assert any("NOT SELF-LOCKING" in w for w in r["warnings"])


def test_the_two_mm_lead_does():
    r = screw.analyse("T8", 2.0, friction=GREASED)
    assert r["self_locking"] is True
    assert r["lead_angle_deg"] == pytest.approx(5.20, abs=0.05)
    assert r["mu_needed_to_lock"] == pytest.approx(0.088, abs=0.002)


def test_the_old_assumed_efficiency_belonged_to_the_other_screw():
    """eta = 0.35 was carried through every force in this design. It is the 2 mm screw's number
    (0.364); the 4 mm runs at 0.52. The design was using 2 mm physics on a 4 mm screw."""
    assert screw.analyse("T8", 2.0, friction=GREASED)["efficiency"] == pytest.approx(0.364, abs=0.01)
    assert screw.analyse("T8", 4.0, friction=GREASED)["efficiency"] == pytest.approx(0.524, abs=0.01)


@pytest.mark.parametrize("lead", [2.0, 4.0, 8.0])
def test_efficiency_and_self_locking_can_never_disagree(lead):
    """eta > 0.5 IS not-self-locking, seen from the other side. A design claiming both is wrong
    about one of them, and the module has to say so rather than let the two coexist."""
    r = screw.analyse("T8", lead, friction=GREASED)
    assert r["self_locking"] is (r["efficiency"] <= 0.5)
    if r["efficiency"] > 0.5:
        assert any("cannot disagree" in w for w in r["warnings"])


def test_a_finer_lead_makes_more_force_from_the_same_motor():
    """The counterintuitive part, and the reason the fix costs nothing but speed: lead and efficiency
    trade against each other, so halving the lead more than compensates for the efficiency drop."""
    f2 = screw.analyse("T8", 2.0, motor_torque_Nm=0.42)["force_N"]
    f4 = screw.analyse("T8", 4.0, motor_torque_Nm=0.42)["force_N"]
    f8 = screw.analyse("T8", 8.0, motor_torque_Nm=0.42)["force_N"]
    assert f2 > f4 > f8
    assert f2 == pytest.approx(480, abs=10)


# ── friction is a band ───────────────────────────────────────────────────────────────────────────

def test_the_verdict_is_taken_at_the_slippery_end():
    """A screw that self-locks only at the high-friction end of its own band does not self-lock.
    Judging at the optimistic end is how the original claim survived."""
    r = screw.analyse("T8", 4.0, friction="steel_brass_dry")     # 0.15-0.19, straddles the 0.176 line
    assert r["self_locking"] is False
    assert r["self_locking_at_best_friction_only"] is True
    assert "not a design" in " ".join(r["warnings"])


def test_vibration_is_a_stricter_test_than_static():
    """This arm rides on a driven chassis. A screw that only just locks on a bench walks down."""
    r = screw.analyse("T8", 2.0, friction=GREASED)
    assert r["self_locking"] is True
    assert r["self_locking_under_vibration"] is False           # 0.088 needed, 0.10 available
    assert any("walks down" in w for w in r["warnings"])


def test_a_much_finer_lead_clears_the_vibration_bar():
    """Guards the test above: if the vibration flag were always False it would prove nothing."""
    assert screw.analyse("T12", 3.0, friction="steel_brass_dry")["self_locking_under_vibration"] is True


# ── backdrive ────────────────────────────────────────────────────────────────────────────────────

def test_backdrive_torque_changes_sign_at_the_self_locking_line():
    """A self-locking screw needs torque APPLIED to lower it. A non-locking one gives torque back --
    which is the arm falling."""
    assert screw.analyse("T8", 2.0, load_N=300)["backdrive_torque_Nm"] > 0
    assert screw.analyse("T8", 4.0, load_N=300)["backdrive_torque_Nm"] < 0


# ── the screw as a column and as a shaft ─────────────────────────────────────────────────────────

def test_an_uncut_screw_buckles_below_the_force_its_own_motor_makes():
    """T8 is sold in 300-500 mm sticks and these rams stroke under 90 mm. Bolt the whole stick in and
    the screw is the weak link: it bows before the motor stalls."""
    r = screw.analyse("T8", 2.0, load_N=480, free_length_mm=400)
    assert r["limited_by"] == "buckling"
    assert r["compression_limit_N"] < 480 * 1.5
    assert any("weak link in compression" in w for w in r["warnings"])


def test_critical_speed_matches_first_principles():
    """This was WRONG, by a factor of 25.4. The published constant 4.76e6 is per INCH and it was
    being fed millimetres, which produced a confident "whips at 164 rpm, inside your operating range"
    that was simply false -- the real answer is about 4100.

    Pinned against the first bending mode of a simply-supported steel shaft, which is what critical
    speed IS: f1 = (pi/2)*sqrt(EI/(rho*A*L^4))."""
    E, rho, d, L = 200e9, 7850.0, 0.0055, 0.400
    I = math.pi * d ** 4 / 64
    A = math.pi * d ** 2 / 4
    rpm = (math.pi / 2) * math.sqrt(E * I / (rho * A * L ** 4)) * 60
    assert screw.analyse("T8", 2.0, free_length_mm=400)["critical_speed_rpm"] ==         pytest.approx(rpm, rel=0.05)


def test_whip_is_NOT_a_reason_to_cut_the_screw():
    """The honest correction. Nothing on this machine turns anywhere near 4100 rpm, so the uncut
    screw does not whirl -- buckling alone justifies cutting it."""
    assert screw.analyse("T8", 2.0, free_length_mm=400)["critical_speed_rpm"] > 3000


def test_cutting_it_to_length_fixes_the_buckling():
    """Same part, one saw cut, ten times the margin. A build instruction, not an analysis."""
    cut = screw.analyse("T8", 2.0, load_N=480, free_length_mm=screw.span_for(64))
    assert cut["limited_by"] == "yield"
    assert cut["compression_limit_N"] > 480 * 5
    assert not any("weak link" in w for w in cut["warnings"])


def test_buckling_matches_euler_by_hand():
    """The other half of the cut-to-length finding, and this one WAS right -- so it gets pinned."""
    E, d, L = 200_000.0, 5.5, 400.0
    I = math.pi * d ** 4 / 64
    assert screw.analyse("T8", 2.0, free_length_mm=L)["buckling_load_N"] ==         pytest.approx(math.pi ** 2 * E * I / L ** 2, rel=0.01)


def test_span_is_stroke_plus_hardware_not_the_length_you_buy():
    assert screw.span_for(64) == pytest.approx(64 + 15 + 30)


# ── refusals ─────────────────────────────────────────────────────────────────────────────────────

def test_an_unknown_thread_is_refused_by_name():
    with pytest.raises(ValueError, match="T8"):
        screw.analyse("M8x1.25")


def test_compare_returns_one_row_per_lead():
    rows = screw.compare("T8", (2.0, 4.0), motor_torque_Nm=0.42)
    assert [r["lead_mm"] for r in rows] == [2.0, 4.0]
    assert rows[0]["self_locking"] and not rows[1]["self_locking"]
