"""power.py — can the supply actually run what is on the bus.

The excavator declared a 2-cell pack (7.4 V nominal, 8.4 V full) and selected a 12 V vacuum pump and
a 12 V solenoid valve. Nothing compared them, because the assumption lived in the build type's
envelope and the choice lived in the BOM — the same shape as the motor torque assumed at 0.42 N·m
while a 0.02 N·m motor sat in the parts list, and the `drive: "awd"` label on a two-motor machine.
Three instances of one missing comparison.

Two questions, both arithmetic:

  **Voltage.** Every part has a voltage it needs. A 12 V pump on a 7.4 V bus does not pull its rated
  vacuum; a 5 V board on 12 V stops being a board. Under-volting is a performance answer, over-volting
  is a destruction answer, and they are reported separately because they are different problems.

  **Current.** The sum of what everything draws against what the supply can give. Steppers dominate
  and they are the honest hard case: rated phase current is not bus current, so the figure here is a
  stated upper bound rather than a measurement, and it says so.

**What it will not do is guess.** A component with no voltage recorded is listed as unknown, not
assumed compatible — a bus verdict computed over the three parts that happened to declare something
would be worth less than no verdict at all.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("mechcalc.power")

# `12V`, `12 V`, `24VDC`, `3.3v`. Deliberately strict: a bare "12" in a part number is not a voltage,
# and reading one out of "MTS-101" would be inventing data rather than finding it.
_V_IN_NAME = re.compile(r"(\d+(?:\.\d+)?)\s*V(?:DC)?\b", re.I)

LIPO_NOMINAL_V = 3.7           # per cell; 4.2 charged, 3.0 empty
UNDER_VOLT_TOL = 0.75          # below three quarters of its rating a part is starved, not merely low
# ONE RULE, SHARED WITH `wiring`. This module allowed a 10% band and called a 12 V pump on an 11.1 V
# rail fine; `wiring` said the same part "needs its own buck/boost converter, which is a part that is
# not in the build yet" -- and both are defensible, which is the problem: two checks reaching
# opposite conclusions about the same two parts is worse than either answer alone. `wiring`'s
# tolerance wins because it names the consequence and the missing part rather than waving it through.
RAIL_MATCH_V = 0.6             # the same band `wiring` uses before it calls for a converter


# `current_cont_a` MEANS DIFFERENT THINGS depending on what the part is, and the first version of
# this module summed them all. A NEMA 17's 1.5 A is what it pulls; a DRV8825's 1.5 A is what it can
# DELIVER; a toggle switch's 6 A is what its contacts are rated to carry. Adding a switch's contact
# rating to a power budget is not a small error -- on this build it was 6 of 12.92 A, and the total
# was handed over as the number to size a pack against.
_DRAWS = ("stepper", "dc motor", "motor", "gearmotor", "servo", "pump", "valve", "solenoid",
          "fan", "heater", "light", "led", "mcu", "controller", "sbc", "camera", "sensor")
_RATINGS = ("motor driver", "esc", "switch", "fuse", "relay", "connector", "wire", "battery",
            "regulator", "converter", "bec", "ubec")


def draws_current(comp: dict) -> tuple[bool, str]:
    """Is this part's `current_cont_a` a DRAW, or a capacity it can carry.

    Whitelisted rather than blacklisted: an unrecognised category is excluded AND said to be
    excluded, because a budget that quietly counts a fuse holder is worse than one that admits it
    left something out.
    """
    cat = str(comp.get("category") or "").strip().lower()
    # RATINGS FIRST, because the categories overlap as SUBSTRINGS and the specific one has to win:
    # "motor driver" contains "motor", so checking loads first counted every stepper driver as a
    # load AND as the thing it drives. The totals happened to agree either way on this build --
    # 6 A of switch removed, 6 A of quantity added -- which is exactly how a wrong number survives.
    if any(k in cat for k in _RATINGS):
        return False, f"{cat}: that figure is what it can carry or deliver, not what it draws"
    if any(k in cat for k in _DRAWS):
        return True, cat
    return False, f"{cat or 'uncategorised'}: not known to be a load, so it is left out of the budget"


def _f(v):
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    return n if n else None


# `voltage_nom_v` CARRIES THREE DIFFERENT MEANINGS and the catalogue has only the one column: a
# REQUIRED supply (the pump's 12 V), a LOGIC level (the ESP32's 3.3 V, regulated on the board), a
# driver's TYPICAL supply that is really a range (the DRV8825 runs 8.2-45 V), and a pack's own rail
# (the 3S LiPo's 11.1 V, which IS the bus it would be compared against).
#
# Comparing all four against the rail produced four "needs a boost converter" findings, of which two
# were nonsense. This is the same lesson `current_cont_a` taught one field earlier: a single column
# holding several physical meanings has to be read per category or not at all.
_RAIL_FED = ("pump", "valve", "solenoid", "motor", "gearmotor", "fan", "heater", "light", "led")
_NOT_A_RAIL_REQUIREMENT = {
    "battery": "it IS the rail, so comparing it against itself says nothing",
    "motor driver": "a driver takes a RANGE (a DRV8825 runs 8.2-45 V); the recorded figure is a "
                    "typical supply, not a requirement",
    "esc": "an ESC takes a range, not a single voltage",
    "mcu": "regulated on the board or by the loom's own converter, which the wiring plan places",
    "controller": "regulated on the board or by the loom's own converter",
    "sbc": "regulated on the board",
    "sensor": "logic-level, fed from the regulated rail rather than the pack",
    "switch": "a contact rating, not a supply",
}


def rail_requirement(comp: dict) -> tuple[bool, str]:
    """Is this part's recorded voltage a SUPPLY IT REQUIRES from the pack rail."""
    cat = str(comp.get("category") or "").strip().lower()
    for key, why in _NOT_A_RAIL_REQUIREMENT.items():
        if key in cat:
            return False, f"{cat}: {why}"
    if any(k in cat for k in _RAIL_FED):
        return True, cat
    return False, f"{cat or 'uncategorised'}: not known to be fed straight off the pack rail"


def needs_volts(comp: dict) -> tuple[float | None, str]:
    """(volts, where it came from). None when the component does not say.

    The field first, the NAME second, nothing third. Every catalogue row on this build has an empty
    `voltage_v`, so a check that read only the field would have had nothing to compare and would have
    passed the very contradiction it exists to catch.
    """
    # `voltage_nom_v` IS THE FIELD. This read `voltage_v`, which no catalogue row has, so it fell
    # through to parsing the name every time -- and got the pump's 12 V only because "12V" happens to
    # be in its name. Fourteen parts were reported "unknown voltage" with the answer sitting in the
    # record. A fallback that works by luck is worse than one that never fires, because it looks like
    # it is working.
    for field in ("voltage_nom_v", "voltage_v", "volts"):
        v = _f(comp.get(field))
        if v:
            return v, f"the catalogue's {field}"
    m = _V_IN_NAME.search(str(comp.get("name") or ""))
    if m:
        try:
            return float(m.group(1)), f"the {m.group(0).strip()} in its name"
        except ValueError:
            pass
    return None, "nothing on the record says what it needs"


def bus_volts(layout: dict | None) -> tuple[float | None, str]:
    """The bus the design declares, and how it was arrived at."""
    layout = layout or {}
    for key in ("bus_v", "supply_v", "voltage_v"):
        v = _f(layout.get(key))
        if v:
            return v, f"declared as {key}"
    cells = _f(layout.get("cells"))
    if cells:
        return cells * LIPO_NOMINAL_V, (f"{cells:g} cells x {LIPO_NOMINAL_V} V nominal "
                                        f"({cells * 4.2:.1f} V charged, {cells * 3.0:.1f} V empty)")
    return None, "the design declares neither a bus voltage nor a cell count"


def check(components: list[dict] | None, layout: dict | None = None) -> dict:
    """Does the declared supply match the parts chosen to run off it.

    {ok, bus_v, under_volted, over_volted, unknown_voltage, current_a, note}. `ok` is False only for
    a real mismatch: an unknown is a gap and is reported as one, because a part that cannot say what
    it needs has not been shown to be compatible OR incompatible.
    """
    bus, bus_why = bus_volts(layout)
    under, over, unknown, current, rated = [], [], [], 0.0, []
    not_a_draw: list = []
    converter: list = []
    not_on_rail: list = []
    for c in components or []:
        spec = c.get("spec") or c
        name = str(c.get("name") or spec.get("name") or "a part")
        v, where = needs_volts(spec if spec.get("name") else c)
        amps = (_f(spec.get("current_cont_a")) or _f(c.get("current_cont_a"))
                or _f(spec.get("current_a")) or _f(c.get("current_a")))
        qty = max(1, int(_f(c.get("qty")) or 1))
        if amps:
            is_draw, cat_why = draws_current(spec if spec.get("category") else c)
            if is_draw:
                current += amps * qty
                rated.append(f"{qty}x {name}" if qty > 1 else name)
            else:
                not_a_draw.append({"part": name, "amps": amps, "why": cat_why})
        if v is None:
            unknown.append({"part": name, "why": where})
            continue
        if bus is None:
            continue
        on_rail, rail_why = rail_requirement(spec if spec.get("category") else c)
        if not on_rail:
            not_on_rail.append({"part": name, "needs_v": v, "why": rail_why})
            continue
        if v > bus / UNDER_VOLT_TOL:
            under.append({"part": name, "needs_v": v, "bus_v": round(bus, 1), "source": where,
                          "why": f"{name} needs {v:g} V and the bus is {bus:.1f} V",
                          "fix": f"raise the pack to at least {v:g} V, or choose a {bus:.1f} V part"})
        elif abs(v - bus) > RAIL_MATCH_V and v > bus:
            # It will run, and not at its rating. A 12 V pump on an 11.1 V nominal rail pulls less
            # vacuum as the pack sags, and the gripper's whole function is the vacuum.
            converter.append({
                "part": name, "needs_v": v, "bus_v": round(bus, 1), "source": where,
                "why": f"{name} wants {v:g} V and the rail is {bus:.1f} V nominal "
                       f"({bus / LIPO_NOMINAL_V * 4.2:.1f} charged, {bus / LIPO_NOMINAL_V * 3.0:.1f} "
                       f"empty), so it is under its rating for most of a charge",
                "fix": f"add a boost converter to {v:g} V for it, or accept it running below rating "
                       f"and say so in the test plan"})
        elif v < bus * UNDER_VOLT_TOL:
            over.append({"part": name, "needs_v": v, "bus_v": round(bus, 1), "source": where,
                         "why": f"{name} is a {v:g} V part on a {bus:.1f} V bus",
                         "fix": f"regulate it down to {v:g} V — a {bus:.0f} V feed will destroy it"})
    out = {
        # `ok` is about parts that CANNOT work. A part that runs below its rating is a
        # decision to take, not a fault to block on, and it is listed either way.
        "ok": not under and not over,
        "bus_v": round(bus, 2) if bus else None, "bus_why": bus_why,
        "under_volted": under, "over_volted": over, "unknown_voltage": unknown,
        "needs_converter": converter, "not_on_the_pack_rail": not_on_rail,
        "current_a": round(current, 2), "current_from": rated,
        "not_a_draw": not_a_draw,
        "note": "voltage is compared per part against the declared bus; a part that does not say "
                "what it needs is listed under `unknown_voltage` rather than assumed to fit",
    }
    if not bus:
        out["ok"] = False
        out["why"] = bus_why
        out["fix"] = "declare `cells` or `bus_v` in the build type's envelope"
    # The current figure is an upper bound and must not be read as a measurement: a stepper's rated
    # phase current is not what it draws from the bus, and only the parts that declare one are in it.
    if rated:
        out["current_note"] = (
            f"{current:.2f} A is what the {len(rated)} LOAD(s) draw, quantity included. "
            f"{len(not_a_draw)} part(s) declare a current that is a capacity or a contact rating "
            f"rather than a draw and are excluded -- see `not_a_draw`. A stepper's rated phase "
            f"current is still not its bus draw, so this remains an upper bound for sizing a supply "
            f"and not a prediction")
    return out


def _selfcheck() -> None:
    # the excavator's real contradiction: a 2S pack against 12 V pneumatics
    comps = [{"name": "12V diaphragm vacuum pump (555-class)", "category": "pump"},
             {"name": "3-way solenoid valve 12V (direct-acting)", "category": "valve"},
             {"name": "NEMA 17 stepper 17HS15-1504S", "category": "stepper",
              "current_cont_a": 1.5}]
    r = check(comps, {"cells": 2})
    assert r["ok"] is False, r
    assert len(r["under_volted"]) == 2, r["under_volted"]
    assert r["bus_v"] == 7.4, r["bus_v"]
    assert len(r["unknown_voltage"]) == 1, "the stepper does not state a voltage and must say so"
    assert r["current_a"] == 1.5

    # a 3S pack runs them
    ok = check(comps, {"cells": 3})
    assert ok["ok"] is True, ok["under_volted"]

    # a 5 V board on a 12 V bus is the other failure, and a different one
    hot = check([{"name": "5V logic board"}], {"bus_v": 12.0})
    assert hot["ok"] is False and len(hot["over_volted"]) == 1

    # no bus declared is a gap, not a pass
    none = check([{"name": "12V pump"}], {})
    assert none["ok"] is False and "declares neither" in none["why"]

    # a part number is not a voltage
    v, _ = needs_volts({"name": "Toggle switch (MTS-101, SPST)"})
    assert v is None, "MTS-101 is a part number, not a 101 V part"
    print(f"power ok — 2S bus {check(comps, {'cells': 2})['bus_v']} V under-volts "
          f"{len(r['under_volted'])} parts; 3S clears them")


if __name__ == "__main__":
    _selfcheck()
