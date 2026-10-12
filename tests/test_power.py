"""Can the supply run what was chosen to hang off it.

The excavator declared a 2-cell pack — 7.4 V nominal — and selected a 12 V vacuum pump and a 12 V
solenoid valve. Nothing compared them, because the assumption lived in the build type's envelope and
the choice lived in the BOM. That is the same missing comparison as the motor torque assumed at
0.42 N·m against a 0.02 N·m motor in the parts list, and the `drive: "awd"` label on a two-motor
machine: three instances of one absent check.
"""
from __future__ import annotations

from mechcalc_mcp import power


def test_the_contradiction_this_module_was_written_for():
    """A 12 V pump and a 12 V valve on a 7.4 V bus. It shipped that way."""
    comps = [{"name": "12V diaphragm vacuum pump (555-class)", "category": "pump"},
             {"name": "3-way solenoid valve 12V (direct-acting)", "category": "valve"}]
    r = power.check(comps, {"cells": 2})
    assert r["ok"] is False
    assert len(r["under_volted"]) == 2
    assert r["bus_v"] == 7.4


def test_three_cells_run_the_same_parts():
    """Guards the test above: if it refused everything it would be noise. 3 x 3.7 = 11.1 V nominal,
    12.6 charged, and a 12 V part is inside the 10% band."""
    comps = [{"name": "12V diaphragm vacuum pump (555-class)", "category": "pump"}]
    r = power.check(comps, {"cells": 3})
    assert r["under_volted"] == [], "11.1 V nominal is not starving a 12 V pump"
    assert len(r["needs_converter"]) == 1, "but it does run under its rating for most of a charge"


def test_over_volting_is_a_different_failure_from_under_volting():
    """Under-volting is a performance answer; over-volting destroys the part. Reported separately
    because the fixes are opposite: raise the pack, or regulate it down."""
    r = power.check([{"name": "5V logic board", "category": "light"}], {"bus_v": 12.0})
    assert r["ok"] is False
    assert len(r["over_volted"]) == 1 and not r["under_volted"]
    assert "regulate it down" in r["over_volted"][0]["fix"]


def test_a_part_number_is_not_a_voltage():
    """`MTS-101` is a toggle switch, not a 101 V part. A pattern loose enough to read a voltage out
    of a part number would invent data, which is worse than the gap it fills."""
    assert power.needs_volts({"name": "Toggle switch (MTS-101, SPST)"})[0] is None
    assert power.needs_volts({"name": "DRV8825 stepper driver carrier"})[0] is None
    assert power.needs_volts({"name": "24VDC solenoid"})[0] == 24.0
    assert power.needs_volts({"name": "3.3 V regulator"})[0] == 3.3


def test_the_field_beats_the_name():
    """Every catalogue row on this build has an empty `voltage_v`, which is why the name is read at
    all — but a recorded number is evidence and a parsed one is a guess, so the field wins."""
    v, where = power.needs_volts({"name": "12V pump", "voltage_v": 24.0})
    assert v == 24.0 and "voltage_v" in where


def test_a_part_that_does_not_say_is_unknown_rather_than_compatible():
    """A bus verdict computed over only the parts that happened to declare something would be worth
    less than no verdict at all."""
    r = power.check([{"name": "DRV8825 stepper driver carrier", "category": "stepper"}],
                    {"cells": 3})
    assert r["unknown_voltage"] and r["unknown_voltage"][0]["part"].startswith("DRV8825")
    assert r["ok"] is True, "an unknown is a gap, not a mismatch — it must not block"


def test_no_declared_bus_is_a_gap_not_a_pass():
    r = power.check([{"name": "12V pump", "category": "pump"}], {})
    assert r["ok"] is False and "declares neither" in r["why"]


def test_the_current_total_says_what_it_is():
    """A stepper's rated phase current is not its bus draw. The number is an upper bound for sizing a
    supply, and calling it a prediction would be the kind of confident wrong figure this codebase
    exists to avoid."""
    r = power.check([{"name": "NEMA 17", "category": "stepper", "current_cont_a": 1.5},
                     {"name": "NEMA 17 b", "category": "stepper", "current_cont_a": 1.5}],
                    {"cells": 3})
    assert r["current_a"] == 3.0
    assert "upper bound" in r["current_note"]


# ── a current budget is a sum of LOADS, not of every number labelled "current" ────────────────────

def test_a_switch_contact_rating_is_not_a_load():
    """The first version of this module summed `current_cont_a` across every part. A NEMA 17's 1.5 A
    is what it pulls; a DRV8825's 1.5 A is what it can DELIVER; a toggle switch's 6 A is what its
    contacts are rated to carry. The switch alone was 6 of a 12.92 A total that was handed over as
    the number to size a pack against."""
    parts = [{"name": "NEMA 17", "category": "stepper", "current_cont_a": 1.5, "qty": 3},
             {"name": "MTS-101", "category": "switch", "current_cont_a": 6.0, "qty": 4}]
    r = power.check(parts, {"cells": 3})
    assert r["current_a"] == 4.5, "three steppers; the switch carries current, it does not draw it"
    assert [x["part"] for x in r["not_a_draw"]] == ["MTS-101"]


def test_a_driver_is_not_counted_as_well_as_the_thing_it_drives():
    """"motor driver" CONTAINS "motor", so checking loads first counted every stepper driver as a
    load and as the thing it drives. The totals happened to agree either way on this build -- 6 A of
    switch removed, 6 A of quantity added -- which is exactly how a wrong number survives a fix."""
    is_draw, why = power.draws_current({"category": "motor driver"})
    assert is_draw is False, "the specific category has to win over the substring"
    assert "deliver" in why
    assert power.draws_current({"category": "stepper"})[0] is True
    assert power.draws_current({"category": "dc motor"})[0] is True


def test_quantity_counts():
    """Three steppers are three. The first version counted one."""
    one = power.check([{"name": "m", "category": "stepper", "current_cont_a": 1.5}], {"cells": 3})
    three = power.check([{"name": "m", "category": "stepper", "current_cont_a": 1.5, "qty": 3}],
                        {"cells": 3})
    assert one["current_a"] == 1.5 and three["current_a"] == 4.5
    assert three["current_from"] == ["3x m"]


def test_an_unrecognised_category_is_left_out_and_says_so():
    """A budget that quietly counts a fuse holder is worse than one that admits it left something
    out, so the whitelist excludes what it does not recognise AND names it."""
    r = power.check([{"name": "mystery", "category": "widget", "current_cont_a": 9.0}], {"cells": 3})
    assert r["current_a"] == 0.0
    assert "not known to be a load" in r["not_a_draw"][0]["why"]


def test_the_voltage_field_the_catalogue_actually_populates():
    """This module read `voltage_v`, which no catalogue row has, and fell through to parsing the
    NAME every time. It got the pump's 12 V only because "12V" happens to be in its name, and
    reported fourteen parts as unknown with the answer sitting in `voltage_nom_v` on the record.

    A fallback that works by luck is worse than one that never fires, because it looks like it is
    working."""
    v, where = power.needs_volts({"name": "Some pump", "voltage_nom_v": 12.0})
    assert v == 12.0 and "voltage_nom_v" in where, "the record beats the name"

    # the name fallback still exists for rows that genuinely have no field
    v2, where2 = power.needs_volts({"name": "12V diaphragm vacuum pump"})
    assert v2 == 12.0 and "name" in where2


def test_a_part_with_neither_field_nor_name_is_still_unknown():
    """The NEMA 17 records `voltage_nom_v: None` -- a stepper's voltage is set by its driver, not by
    the motor -- so it stays honestly unknown rather than being given a number."""
    assert power.needs_volts({"name": "NEMA 17 stepper 17HS15-1504S", "voltage_nom_v": None})[0] is None
