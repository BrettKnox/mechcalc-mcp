"""Ram geometry, reach, and the stability envelope the old model could not see.

Three numbers the design asserted:

- the ram's moment arm, taken as a constant 40 mm when it is 40 mm only at its best pose and 10 mm
  at its worst;
- reach, typed in as 320 mm when the elbow's own limit stops the arm at 278;
- tip-over, computed as `(track/2)/cog_height` for a vehicle standing still, on a machine whose
  whole job is to swing a mass out over its own wheels.

The last one was not conservative. It was 1.64 g against a real 0.14 g, and the build type demands
0.60.
"""
from __future__ import annotations

import math

import pytest

from mechcalc import mechanism as M

ARM = dict(wheelbase_mm=220, cog_height_mm=55, base_kg=2.6, arm_kg=1.23,
           payload_kg=0.5, reach_mm=320)


# ── the ram ──────────────────────────────────────────────────────────────────────────────────────

def test_the_moment_arm_is_not_constant():
    """It is 2*area/base through the clevis triangle, and it varies by 3x across this joint's range.
    Sizing the actuator from one pose is sizing it from the wrong pose."""
    r = M.ram(40, 40, 55, 150, force_N=480)
    assert r["moment_arm_max_mm"] > 3 * r["moment_arm_min_mm"]


def test_the_effective_arm_never_reaches_the_nominal_one():
    """For equal arms the geometry collapses to h = a*cos(theta/2), so the moment arm is at most `a`
    and only at zero degrees -- which is the toggle. Across a usable range it is always less.

    The design computed its reduction from a flat 40 mm. The best pose in range gives 35.5 mm and the
    worst gives 10.4, so the quoted reduction was optimistic everywhere and 4x optimistic at the end
    of the stroke."""
    r = M.ram(40, 40, 55, 150)
    assert r["moment_arm_max_mm"] < 40.0
    assert r["moment_arm_max_mm"] == pytest.approx(40 * math.cos(math.radians(55 / 2)), abs=0.1)
    assert r["moment_arm_min_mm"] == pytest.approx(40 * math.cos(math.radians(150 / 2)), abs=0.1)


def test_a_toggle_point_inside_the_travel_is_refused_loudly():
    """At 0 and 180 degrees the ram line passes through the pivot: infinite force, zero torque. A
    range spanning either has a pose where the joint cannot be moved at all."""
    r = M.ram(40, 40, 5, 175, force_N=480)
    assert any("TOGGLE" in w for w in r["warnings"])
    assert r["moment_arm_min_mm"] < 0.1 * r["moment_arm_max_mm"]


def test_a_sensible_range_has_no_toggle():
    """Guards the test above."""
    assert not any("TOGGLE" in w for w in M.ram(40, 40, 55, 150, force_N=480)["warnings"])


def test_the_worst_margin_is_found_by_sweeping_not_assumed():
    r = M.ram(40, 40, 55, 150, force_N=480, load_torque_Nm=1.7)
    assert r["margin_min"] > 1.0
    assert r["range_deg"][0] <= r["margin_worst_angle_deg"] <= r["range_deg"][1]


def test_a_ram_that_stalls_inside_its_own_range_is_named():
    """It drives to the tight spot and stops, which reads as a seized joint rather than a sizing
    error unless something says so."""
    r = M.ram(40, 40, 55, 150, force_N=20, load_torque_Nm=1.7)
    assert r["margin_min"] < 1.0
    assert any("STALLS INSIDE" in w for w in r["warnings"])


def test_stroke_is_derived_from_the_geometry():
    r = M.ram(40, 40, 55, 150)
    assert r["stroke_mm"] == pytest.approx(r["ram_length_max_mm"] - r["ram_length_min_mm"], abs=0.01)


@pytest.mark.parametrize("a,b", [(0, 40), (40, 0), (-5, 40)])
def test_nonpositive_arms_are_refused(a, b):
    with pytest.raises(ValueError, match="positive"):
        M.ram(a, b, 55, 150)


def test_a_zero_width_range_is_refused():
    with pytest.raises(ValueError, match="more than zero"):
        M.ram(40, 40, 90, 90)


# ── workspace ────────────────────────────────────────────────────────────────────────────────────

def test_the_elbow_limit_shortens_the_reach():
    """320 mm was declared. 180 + 140 only straightens to 320 if the elbow opens to 180 degrees,
    and this one opens to 120."""
    w = M.workspace(180, 140, joint2_deg=120)
    assert w["reach_if_straight_mm"] == pytest.approx(320.0, abs=0.1)
    assert w["reach_mm"] < 300
    assert any("rather than" in n for n in w["notes"])


def test_an_elbow_that_straightens_reaches_further():
    """Guards the test above: if reach ignored the limit, both would be equal."""
    assert M.workspace(180, 140, joint2_deg=180)["reach_mm"] > \
           M.workspace(180, 140, joint2_deg=120)["reach_mm"]


def test_a_tool_adds_to_the_reach():
    assert M.workspace(180, 140, tool_mm=60)["reach_mm"] > M.workspace(180, 140)["reach_mm"]


# ── tip-over ─────────────────────────────────────────────────────────────────────────────────────

def test_stowed_it_reproduces_the_old_static_formula():
    """The test that this GENERALISES the previous answer rather than replacing it with a different
    guess. With the arm in, the two must agree."""
    r = M.tipover(track_mm=180, **{**ARM, "reach_mm": 0, "payload_kg": 0})
    assert r["worst_tip_g"] == pytest.approx(r["static_tip_g"], abs=0.02)


def test_reaching_out_is_dramatically_worse_and_the_old_number_hid_it():
    """1.64 g parked, 0.14 g reaching sideways loaded. The static figure flatters the machine by 12x
    and is the only number the design had."""
    r = M.tipover(track_mm=180, **ARM)
    assert r["static_tip_g"] > 1.5
    assert r["worst_tip_g"] < 0.2
    assert any("flatters" in w for w in r["warnings"])


def test_the_design_as_drawn_fails_its_own_feasibility_rule():
    """excavator_base declares tip_over_lateral_g >= 0.60."""
    assert M.tipover(track_mm=180, **ARM)["worst_tip_g"] < 0.60


def test_a_square_footprint_plus_a_counterweight_fixes_it():
    """Track 220 to match the 220 wheelbase makes the support polygon square, so no heading is worse
    than any other; 0.3 kg on the house does the rest."""
    r = M.tipover(track_mm=220, counterweight_kg=0.3, counterweight_r_mm=100, **ARM)
    assert r["worst_tip_g"] >= 0.60


def test_the_counterweight_swings_with_the_arm():
    """It belongs to the HOUSE. Bolted to the base instead it would help in one heading and hurt in
    the opposite one, which is worse than not having it -- and that is what this model did first."""
    plain = M.tipover(track_mm=220, **ARM)["worst_tip_g"]
    with_cw = M.tipover(track_mm=220, counterweight_kg=0.5, counterweight_r_mm=100, **ARM)["worst_tip_g"]
    assert with_cw > plain


def test_widening_past_the_wheelbase_buys_nothing():
    """Once the track exceeds the wheelbase the short axis becomes the limit, so the extra width is
    wasted. Worth knowing before cutting longer axles than the machine can use."""
    at220 = M.tipover(track_mm=220, **ARM)["worst_tip_g"]
    at300 = M.tipover(track_mm=300, **ARM)["worst_tip_g"]
    assert at300 == pytest.approx(at220, abs=0.01)


def test_a_machine_that_tips_under_its_own_weight_says_so():
    r = M.tipover(track_mm=120, wheelbase_mm=120, cog_height_mm=55,
                  base_kg=1.0, arm_kg=1.2, payload_kg=0.5, reach_mm=400)
    assert r["worst_tip_g"] <= 0
    assert any("TIPS OVER ON ITS OWN" in w for w in r["warnings"])


def test_a_zero_cog_height_is_refused():
    with pytest.raises(ValueError, match="positive"):
        M.tipover(track_mm=180, wheelbase_mm=220, cog_height_mm=0, base_kg=2.6, arm_kg=1.2)


# ── gear mesh ────────────────────────────────────────────────────────────────────────────────────

def test_an_internal_pair_is_a_difference_not_a_sum():
    """The classic error. (108 + 27)/2 = 67.5 puts the pinion outside the ring entirely; the right
    answer is (108 - 27)/2 = 40.5."""
    assert M.gear_mesh(1.5, 72, 18, internal=True)["centre_distance_mm"] == 40.5
    assert M.gear_mesh(1.5, 72, 18, internal=False)["centre_distance_mm"] == 67.5


def test_the_slew_pair_as_shipped_meshes():
    """It did not. The build type placed the pinion at 45 mm: right module, right tooth counts, right
    ratio, both gears printing and measuring correctly, and the teeth simply did not touch. The same
    shape of failure as the planetary, one level up."""
    assert M.gear_mesh(1.5, 72, 18, internal=True, placed_centre_mm=40.5)["meshes"] is True


def test_a_pair_placed_wrong_says_so_with_both_numbers():
    bad = M.gear_mesh(1.5, 72, 18, internal=True, placed_centre_mm=45.0)
    assert bad["meshes"] is False
    assert bad["error_mm"] == pytest.approx(4.5)
    assert any("DO NOT MESH" in w and "40.5" in w for w in bad["warnings"])
    assert any("never touch" in w for w in bad["warnings"])


def test_too_close_is_named_differently_from_too_far():
    """They fail in opposite ways and the fix is opposite too."""
    close = M.gear_mesh(1.5, 72, 18, internal=True, placed_centre_mm=36.0)
    assert any("jam or interfere" in w for w in close["warnings"])


def test_the_wheels_spin_before_the_motors_stall():
    """The question was "how much torque to move it with the added weight". The answer is that
    torque is not what binds: 0.42 N.m through 20:1 on four wheels is about 790 N of push against a
    friction ceiling near 23 N. Adding a kilo changes the demand by a couple of newtons and the
    limit not at all."""
    r = M.drive(mass_kg=2.6, wheel_dia_mm=85.0, crr=0.03, mu=0.9, gear_ratio=20.0,
                        motor_stall_Nm=0.42, driven_wheels=4, grade_deg=10.0)
    assert r["limited_by"] == "traction"
    assert r["margin_x"] > 50
    assert any("traction-limited" in w for w in r["warnings"])


def test_more_mass_costs_force_but_not_the_verdict():
    light = M.drive(mass_kg=1.35, wheel_dia_mm=85.0, gear_ratio=20.0, motor_stall_Nm=0.42)
    heavy = M.drive(mass_kg=3.2, wheel_dia_mm=85.0, gear_ratio=20.0, motor_stall_Nm=0.42)
    assert heavy["force_N"]["total"] > light["force_N"]["total"]
    assert heavy["torque_per_motor_Nm"] > light["torque_per_motor_Nm"]
    assert heavy["limited_by"] == light["limited_by"] == "traction"     # grip scales with weight too


def test_a_motor_that_cannot_climb_is_told_so():
    """Guards the test above: if the verdict were always "traction" it would prove nothing."""
    r = M.drive(mass_kg=2.6, wheel_dia_mm=85.0, gear_ratio=1.0, motor_stall_Nm=0.005,
                        grade_deg=10.0)
    assert r["limited_by"] == "motor torque"
    assert any("will not climb" in w for w in r["warnings"])


def test_the_grade_is_most_of_the_demand_not_the_rolling():
    """Flat ground flatters a drive. On 10 degrees the climb is several times the rolling term, and
    a machine sized on the level meets the lip of a tray and stops."""
    f = M.drive(mass_kg=2.6, wheel_dia_mm=85.0, grade_deg=10.0)["force_N"]
    assert f["grade"] > 4 * f["rolling"]
    flat = M.drive(mass_kg=2.6, wheel_dia_mm=85.0, grade_deg=0.0)["force_N"]
    assert flat["grade"] == 0.0


def test_traction_falls_on_a_slope_because_the_normal_force_does():
    level = M.drive(mass_kg=2.6, wheel_dia_mm=85.0, grade_deg=0.0)["traction_ceiling_N"]
    hill = M.drive(mass_kg=2.6, wheel_dia_mm=85.0, grade_deg=30.0)["traction_ceiling_N"]
    # both are reported to 0.1 N, so compare at that resolution rather than to floats
    assert hill == pytest.approx(level * math.cos(math.radians(30)), abs=0.1)


def test_it_slides_before_it_pitches_and_says_which():
    """A 220 mm wheelbase against a 55 mm CoG tips at 2 g, and the tyres let go at 0.9 g, so this
    one slides. A tall narrow build gets the opposite answer and needs to hear it."""
    ok = M.drive(mass_kg=2.6, wheel_dia_mm=85.0, mu=0.9, cog_height_mm=55.0,
                         wheelbase_mm=220.0)
    assert ok["tip_forward_at_ms2"] == pytest.approx(19.62, abs=0.1)
    assert not any("pitches before it slides" in w for w in ok["warnings"])
    tall = M.drive(mass_kg=2.6, wheel_dia_mm=85.0, mu=0.9, cog_height_mm=200.0,
                           wheelbase_mm=220.0)
    assert any("pitches before it slides" in w for w in tall["warnings"])


@pytest.mark.parametrize("bad", [{"mass_kg": 0.0}, {"wheel_dia_mm": 0.0}])
def test_it_refuses_nonsense_rather_than_dividing_by_it(bad):
    kw = {"mass_kg": 2.6, "wheel_dia_mm": 85.0, **bad}
    with pytest.raises(ValueError):
        M.drive(**kw)
