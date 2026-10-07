"""A planetary gearset is four rules that must hold at once, and three of them are silent.

Asked for a print that tests the system rather than a flat plate. This is the arithmetic behind it,
and the reason it is worth having: the first combination tried by hand — sun 16, planet 16, ring 48,
three planets — meshes perfectly and cannot be assembled.
"""
from __future__ import annotations

from mechcalc import planetary


def test_the_combination_that_meshes_and_still_cannot_be_built():
    """48 = 16 + 2x16, so the teeth are right. (16+48) is 64, which is not divisible by 3, so the
    third planet has nowhere to sit. Nothing in the geometry says so."""
    bad = planetary.design(1.5, 16, 16, 3)
    assert bad["ring_teeth"] == 48, "the teeth do fit; that is what makes it a trap"
    assert not bad["ok"]
    assert "not divisible by 3" in bad["problems"][0]
    assert "64" in bad["problems"][0], "a refusal without the number is not actionable"
    # and it names what would work
    assert "2" in bad["problems"][0] and "4" in bad["problems"][0]


def test_a_sound_set_derives_everything_the_parts_need():
    d = planetary.design(1.5, 18, 18, 3)
    assert d["ok"] and not d["warnings"]
    assert d["ring_teeth"] == 54
    assert d["ratio"] == 4.0
    assert d["centre_distance_mm"] == 27.0
    # the carrier's pin circle IS twice the centre distance; if these ever disagree the printed
    # carrier holds the planets somewhere the gears are not
    assert d["carrier_pin_circle_mm"] == 2 * d["centre_distance_mm"]
    # and the set closes: sun radius + planet diameter == ring pitch radius
    assert d["sun_pitch_d_mm"] / 2 + d["planet_pitch_d_mm"] == d["ring_pitch_d_mm"] / 2


def test_planets_that_would_foul_each_other_are_refused_with_the_gap():
    crowded = planetary.design(1.5, 18, 18, 8)
    assert not crowded["ok"]
    assert any("foul" in p and "mm" in p for p in crowded["problems"])


def test_an_undercut_gear_warns_rather_than_blocking():
    """Twelve teeth at 20 degrees undercuts, and still turns. Refusing it would be wrong; saying
    nothing would be worse."""
    small = planetary.design(1.0, 12, 12, 3)
    assert small["ok"] and small["warnings"]
    assert "undercut" in small["warnings"][0]


def test_suggest_only_offers_sets_that_pass_every_rule():
    picks = planetary.suggest(1.5, 3, want_ratio=4.0)
    assert picks
    for d in picks:
        assert d["ok"] and not d["warnings"]
        assert (d["sun_teeth"] + d["ring_teeth"]) % d["n_planets"] == 0
    assert abs(picks[0]["ratio"] - 4.0) < 0.01, "the closest ratio should be offered first"
