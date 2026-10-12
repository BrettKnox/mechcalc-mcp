"""The arm engine counted every motor in total mass and none of them in the shoulder torque, the tip
deflection or the root stress. A servo hung 280 mm out on a wrist loaded the shoulder for free, so the
engine passed an actuator that a hand moment balance showed could not hold the arm out."""
from mechcalc_mcp.engines import _common, arm

G = _common.G
BASE = {"payload_g": 0.0, "link1_mm": 150.0, "link2_mm": 150.0, "motor_mass_g": 60.0}


def _m(**over):
    return arm.analyze({**BASE, **over})["metrics"]


def test_a_distal_mass_adds_exactly_its_own_moment():
    bare = _m(distal_masses=[])
    loaded = _m(distal_masses=[[1000.0, 300.0]])          # 1 kg at 300 mm -> 2.943 N*m
    assert abs(loaded["shoulder_torque_Nm"] - bare["shoulder_torque_Nm"] - 1.0 * G * 0.300) < 0.011


def test_default_places_the_elbow_motor_at_the_elbow_and_wrist_motors_at_the_tip():
    bare = _m(distal_masses=[])["shoulder_torque_Nm"]
    elbow = 0.060 * G * 0.150
    wrist = 0.060 * G * 0.300
    assert abs(_m(dof=2)["shoulder_torque_Nm"] - bare) < 0.011                 # turret: nothing distal
    assert abs(_m(dof=3)["shoulder_torque_Nm"] - bare - elbow) < 0.011
    assert abs(_m(dof=5)["shoulder_torque_Nm"] - bare - elbow - 2 * wrist) < 0.011


def test_root_stress_carries_the_same_moment_as_the_shoulder():
    # a real payload in both, so two-decimal rounding of the metrics cannot swamp the ratio
    a, b = _m(payload_g=500.0, distal_masses=[]), _m(payload_g=500.0, distal_masses=[[500.0, 300.0]])
    ratio = b["shoulder_torque_Nm"] / a["shoulder_torque_Nm"]
    assert abs(b["root_stress_mpa"] / a["root_stress_mpa"] - ratio) < 0.02


def test_a_distal_mass_at_the_tip_deflects_like_a_tip_load():
    d = {**BASE, "link_od_mm": 20.0, "link_wall_mm": 2.0}
    curves = arm.analyze({**d, "distal_masses": []})["_curves"]
    extra = 0.5 * G * 300.0 ** 3 / (3 * curves["E"] * curves["I"])          # 0.5 kg at the 300 mm tip
    a = arm.analyze({**d, "distal_masses": []})["metrics"]["tip_deflection_mm"]
    b = arm.analyze({**d, "distal_masses": [[500.0, 300.0]]})["metrics"]["tip_deflection_mm"]
    assert abs(b - a - extra) < 0.011


def test_total_mass_is_unchanged_when_no_list_is_given():
    m = arm.analyze({**BASE, "dof": 4})
    link = m["_curves"]["link_mass"]
    assert abs(m["metrics"]["mass_kg"] - (link + 4 * 0.060)) < 0.006


def test_radii_past_the_reach_are_clamped_to_it():
    assert _m(distal_masses=[[100.0, 9999.0]]) == _m(distal_masses=[[100.0, 300.0]])
