"""Every tool answers through the MCP layer, and bad input comes back as an error, not a crash."""
import asyncio
import json

from mechcalc_mcp.engines import ENGINES
from mechcalc_mcp.server import mcp

CALLS = {
    "materials_list": {},
    "material_get": {"name": "petg"},
    "engine_list": {},
    "engine_analyze": {"engine": "arm", "design": {"reach_mm": 380, "payload_g": 250,
                                                  "distal_masses": [[55, 280], [55, 330]]}},
    "leadscrew": {"thread": "T8", "lead_mm": 8, "load_N": 50, "motor_torque_Nm": 0.4, "free_length_mm": 300},
    "leadscrew_compare": {"leads": [2, 8]},
    "lever_actuator": {"anchor_mm": 120, "boss_mm": 40, "angle_from_deg": 0, "angle_to_deg": 90, "force_N": 200},
    "arm_workspace": {"link1_mm": 200, "link2_mm": 180},
    "tipover": {"track_mm": 300, "wheelbase_mm": 300, "cog_height_mm": 120, "base_kg": 4, "arm_kg": 1,
                "payload_kg": 0.25, "reach_mm": 380},
    "drive_sizing": {"mass_kg": 6, "wheel_dia_mm": 100, "motor_stall_Nm": 2.9, "driven_wheels": 3},
    "gear_mesh": {"module_mm": 1.5, "teeth_a": 18, "teeth_b": 36, "placed_centre_mm": 40.5},
    "planetary_design": {"module": 1.5, "sun": 18, "planet": 18},
    "planetary_suggest": {"want_ratio": 5},
    "power_check": {"components": [{"name": "STS3215", "category": "servo", "voltage_nom_v": 12,
                                    "current_cont_a": 0.5, "qty": 6}], "layout": {"cells": 3}},
    "mass_aggregate": {"items": [{"name": "a", "mass_g": 100, "x_mm": 0, "y_mm": 0, "z_mm": 0},
                                 {"name": "b", "mass_g": 100, "x_mm": 100, "y_mm": 0, "z_mm": 0}]},
    "battery_mass": {"capacity_mah": 2200, "v_nom": 11.1},
    "print_hole": {"nominal_mm": 3.0},
    "print_shaft": {"nominal_mm": 3.0},
    "print_insert_boss": {"screw_size": "M3"},
    "print_wall": {"target_mm": 2.0},
    "calibrate_plan": {"material": "pla"},
    "fea_beam": {"length": 100, "width": 10, "height": 10, "load": 100},
    "fea_column": {"length": 500, "width": 5, "height": 5},
    "fea_bolt": {"torque_nm": 1.31, "diameter_mm": 3},
    "fea_thermal": {"length_mm": 200, "delta_t": 40, "material": "pla", "constrained": True},
    "wearable_pressure": {"force_n": 40, "pad_area_cm2": 30},
    "wearable_capacity": {"pad_area_cm2": 30},
}
WRITES = {"calibrate_record_hole"}   # covered by tests/test_dfm.py against a temp store


def _call(name, args):
    r = asyncio.run(mcp.call_tool(name, args))
    assert not r.is_error, (name, r.content)
    if r.structured_content is not None:
        return r.structured_content.get("result", r.structured_content)
    out = [json.loads(c.text) for c in r.content]
    return out[0] if len(out) == 1 else out


def test_every_tool_is_exercised():
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == set(CALLS) | WRITES


def test_every_tool_answers():
    for name, args in CALLS.items():
        out = _call(name, args)
        json.dumps(out)
        assert out, name
        if isinstance(out, dict):
            assert out.get("ok", True) is not False, (name, out)


def test_arm_counts_distal_masses():
    bare = _call("engine_analyze", {"engine": "arm", "design": {"reach_mm": 380}})
    loaded = _call("engine_analyze", {"engine": "arm", "design": {"reach_mm": 380,
                                                                  "distal_masses": [[200, 380]]}})
    assert loaded["metrics"]["shoulder_torque_Nm"] > bare["metrics"]["shoulder_torque_Nm"]


def test_every_engine_runs_on_defaults_without_numpy():
    for name in ENGINES:
        out = _call("engine_analyze", {"engine": name})
        assert out["ok"] and out["metrics"], name


def test_bad_input_is_an_error_not_a_crash():
    assert _call("material_get", {"name": "unobtainium"})["ok"] is False
    assert _call("engine_analyze", {"engine": "teleporter"})["ok"] is False
    assert _call("fea_beam", {"length": -1, "width": 1, "height": 1, "load": 1})["ok"] is False
