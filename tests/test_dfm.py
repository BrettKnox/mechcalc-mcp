"""The difference between a model that is right and a part that fits.

Printed holes shrink and heat-set inserts need a specific pilot. Leaving either nominal is worse than
not knowing, because the numbers look authoritative and the parts do not fit.
"""
from __future__ import annotations

import pytest

from mechcalc_mcp import dfm


def test_a_hole_is_modelled_oversize_so_the_printed_one_is_nominal():
    """The compensation is `dfm`'s own constant and `calibrate` exists to REPLACE it with a measured
    one. Asserting 4.4 pins the guess: recalibrating the printer — the thing this module is built to
    make possible — would fail a test that is still doing its job. The invariant is the direction
    and the size of the correction, taken from the module that owns it."""
    h = dfm.hole(4.2)
    assert h["model_mm"] > 4.2, "no compensation means an M4 pin will not enter a 4.2 mm bore"
    assert h["model_mm"] == pytest.approx(4.2 + dfm.ASSUMED_HOLE_SHRINK_MM, abs=1e-6)
    assert h["source"] == "assumed" and "coupon" in h["note"], \
        "an uncalibrated number must say that it is a guess"


def test_a_measured_shrink_beats_the_default_and_says_so():
    m = dfm.hole(4.2, measured_shrink_mm=0.31)
    assert m["model_mm"] == pytest.approx(4.2 + 0.31, abs=1e-6) and m["source"] == "measured"
    assert m["model_mm"] != dfm.hole(4.2)["model_mm"], \
        "if the measured value matched the assumption this test could not tell them apart"
    assert not m["note"], "a measured value should not carry the go-and-measure-it warning"


def test_a_hole_printed_on_its_side_gets_more():
    """It has no floor to sit on and the top sags inward."""
    assert dfm.hole(4.2, axis="x")["model_mm"] > dfm.hole(4.2, axis="z")["model_mm"]


def test_an_insert_hole_is_compensated_to_land_on_the_datasheet_pilot():
    """An insert forms its own THREAD; it does not choose its own PILOT.

    This used to assert the opposite -- that the hole stays nominal because the insert melts its own
    fit. That confuses the two. A hole modelled at the datasheet's 4.0 prints nearer 3.8, and the
    insert then displaces the difference: molten PLA is not compressible, so it flashes onto the
    seating face (the part no longer sits flat) and runs into the insert's own thread (the screw
    seizes). Getting it in needs more force and a longer dwell at 210-230 C, which degrades the PLA
    around it.
    """
    b = dfm.insert_boss("M3")
    want = dfm.FASTENERS["M3"]["insert_hole"]
    assert b["printed_mm"] == want, "the pilot is what must come out of the printer"
    assert b["hole_mm"] > want, "so the modelled hole is larger by the shrink"
    assert b["hole_mm"] == pytest.approx(want + dfm.ASSUMED_HOLE_SHRINK_MM, abs=1e-6)
    assert b["boss_od_mm"] >= want + 2 * dfm.INSERT_WALL_MM, \
        f"{dfm.INSERT_WALL_MM} mm of wall each side, or the boss splits"
    assert not dfm.insert_boss("M99")["ok"]


def test_an_insert_hole_gets_a_lead_in_and_somewhere_for_the_plastic_to_go():
    """The chamfer starts it square; the relief under it takes what the knurls displace instead of
    pushing it up into the thread."""
    b = dfm.insert_boss("M4")
    assert b["chamfer_mm"] > 0
    assert b["relief_dia_mm"] > b["printed_mm"]
    assert "flashes" in b["compensated"]


def test_a_measured_shrink_moves_the_insert_hole_too():
    """It is the same shrink as every other hole -- so a printer that has been calibrated should
    stop guessing here as well."""
    loose = dfm.insert_boss("M3", measured_shrink_mm=0.30)
    assert loose["hole_mm"] == pytest.approx(4.30, abs=0.01)
    assert loose["printed_mm"] == 4.0


def test_a_wall_is_a_whole_number_of_beads():
    """1.5 mm on a 0.4 nozzle is three beads and a gap the slicer cannot lay.

    Asserted as the RULE — a whole number of beads, and the nearest one to what was asked — rather
    than as 4 x 0.42. The extrusion width is a shop setting: changing the nozzle is a legitimate
    change that must not fail a test of the rounding.
    """
    w = dfm.wall(1.5)
    ew = w["extrusion_width_mm"]
    assert w["beads"] >= 1 and w["wall_mm"] == pytest.approx(w["beads"] * ew, abs=0.005)
    assert abs(w["wall_mm"] - 1.5) <= ew / 2 + 0.005, "not the nearest whole number of beads"
    assert dfm.wall(1.5, nozzle_mm=0.6)["extrusion_width_mm"] > ew, "the nozzle sets the bead"


def test_the_checklist_names_inserts_for_screws_into_plastic():
    notes = dfm.review({"name": "bracket", "bbox_mm": [40, 40, 10]})["notes"]
    assert any("insert" in n["fix"] for n in notes), "tapping PLA is the mistake worth naming"


def test_every_measured_bore_is_kept_not_just_the_last(tmp_path, monkeypatch):
    """One bore is one number with no way to know if it is representative. A set across diameters
    shows whether the shrink is a constant, which decides whether one compensation fits every hole."""
    from mechcalc_mcp import calibrate
    monkeypatch.setattr(calibrate, "STORE", tmp_path / "cal.json")
    calibrate.record_hole("pla", 3.6, 3.42)
    calibrate.record_hole("pla", 8.2, 8.05)
    rec = calibrate.record_hole("pla", 4.5, 4.31)
    assert rec["hole_shrink_n"] == 3
    assert [p["nominal_mm"] for p in rec["hole_points"]] == [3.6, 4.5, 8.2], "kept, and sorted"
    # the mean of the points, to well inside what a caliper can read
    assert abs(rec["hole_shrink_mm"] - (0.18 + 0.15 + 0.19) / 3) < 1e-3

    # re-measuring the same bore corrects it rather than voting twice
    again = calibrate.record_hole("pla", 4.5, 4.35)
    assert again["hole_shrink_n"] == 3
    assert [p["measured_mm"] for p in again["hole_points"] if p["nominal_mm"] == 4.5] == [4.35]


def test_points_that_disagree_say_so_instead_of_averaging_it_away(tmp_path, monkeypatch):
    from mechcalc_mcp import calibrate
    monkeypatch.setattr(calibrate, "STORE", tmp_path / "cal.json")
    calibrate.record_hole("pla", 3.6, 3.55)      # 0.05
    rec = calibrate.record_hole("pla", 8.2, 7.80)  # 0.40 — nothing like the first
    assert rec["hole_shrink_spread_mm"] > abs(rec["hole_shrink_mm"])
    assert "not one constant" in rec["hole_shrink_note"]


def test_a_peg_and_a_hole_are_compensated_in_OPPOSITE_directions():
    """The classic printed-fit failure, and the reason it is a failure rather than a near miss.

    Same bead width, opposite sign: a hole shrinks and a peg grows. Draw both nominal at 4.3 and you
    get a 4.5 mm pin going into a 4.1 mm hole — the two errors ADD. Compensated, the pin is drawn
    under and the hole over, and they meet.
    """
    bore = dfm.hole(4.3)["model_mm"]
    pin = dfm.shaft(4.3, clearance_mm=0.25)["model_mm"]
    assert bore > 4.3, "a hole must be drawn OVER its target"
    assert pin < 4.3, "a peg must be drawn UNDER its target"
    assert bore - pin > 0.4, "there is no fit between them"


def test_a_negative_clearance_is_an_interference_and_says_so():
    """A spigot that has to stay put is drawn fatter, not thinner. It should not claim to turn."""
    press = dfm.shaft(4.5, clearance_mm=-0.05)
    free = dfm.shaft(4.5, clearance_mm=0.25)
    assert press["model_mm"] > free["model_mm"]
    assert "interference" in press["why"] and "grips" in press["why"]
    assert "turns freely" not in press["why"]


def test_a_shaft_too_small_to_compensate_is_refused_not_made_negative():
    assert not dfm.shaft(0.1)["ok"]


