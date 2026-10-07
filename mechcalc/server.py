"""MCP server: deterministic engineering checks for robots and 3D-printed hardware.

Every tool is closed-form or table-driven and runs locally with no network and no model calls, so an
agent can design a part and get a number it can trust in the same loop. Results say what they did not
check (`assumptions`, `warnings`). Run: `mechcalc-mcp` (stdio).
"""
from __future__ import annotations

import dataclasses
import functools

from mcp.server.mcpserver import MCPServer

from . import calibrate, dfm, fea, massprops, matprops, mechanism, planetary, power, screw, wearable
from .engines import ENGINES

mcp = MCPServer("mechcalc", instructions=__doc__)


def _safe(fn):
    """A bad input comes back as {"ok": false, "error": ...} instead of killing the call."""
    @functools.wraps(fn)
    def wrap(*a, **kw):
        try:
            return fn(*a, **kw)
        except (ValueError, TypeError, KeyError, matprops.UnknownMaterial) as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return wrap


def tool(fn):
    return mcp.tool()(_safe(fn))


# ── materials ────────────────────────────────────────────────────────────────────────────────────
@tool
def materials_list() -> dict:
    """Material keys and which are FDM-printable. Every other tool takes one of these names."""
    return {"ok": True, "materials": matprops.keys(), "printable": matprops.printable()}


@tool
def material_get(name: str) -> dict:
    """Full property record (E, density, allowable, CTE, layer strength ratio, source). Measured
    coupon data from calibrate_* overrides the datasheet value and says so in `source`."""
    return {"ok": True, **dataclasses.asdict(matprops.get(name))}


# ── system-level engines ─────────────────────────────────────────────────────────────────────────
@tool
def engine_list() -> dict:
    """The archetype physics engines and each one's input keys with defaults. Pass any subset of
    those keys to engine_analyze; the rest keep their defaults."""
    return {"ok": True, "engines": {k: {"about": (m.__doc__ or "").strip().splitlines()[0],
                                        "defaults": m.DEFAULTS} for k, m in ENGINES.items()}}


@tool
def engine_analyze(engine: str, design: dict | None = None) -> dict:
    """First-principles check of a whole machine: arm, vehicle, cnc, multirotor, legged, humanoid,
    linear, balance, vacuum. Returns metrics (torque margins, deflection, safety factors, modes) and
    warnings. For an arm, pass distal_masses=[[mass_g, radius_mm], ...] for every motor or tool past
    the shoulder so shoulder torque counts them."""
    mod = ENGINES.get(engine)
    if mod is None:
        return {"ok": False, "error": f"unknown engine {engine!r}", "have": sorted(ENGINES)}
    r = mod.analyze(design or {})
    return {"ok": True, **{k: v for k, v in r.items() if not k.startswith("_")}}


# ── mechanisms ───────────────────────────────────────────────────────────────────────────────────
@tool
def leadscrew(thread: str = "T8", lead_mm: float = 2.0, friction: str = screw.DEFAULT_FRICTION,
              load_N: float = 0.0, motor_torque_Nm: float = 0.0, free_length_mm: float = 0.0,
              end: str = "pinned") -> dict:
    """Leadscrew efficiency band, back-drive, thrust from motor torque, buckling and whip speed."""
    return screw.analyse(thread, lead_mm, friction=friction, load_N=load_N, motor_torque_Nm=motor_torque_Nm,
                         free_length_mm=free_length_mm, end=end)


@tool
def leadscrew_compare(thread: str = "T8", leads: list[float] = (2.0, 4.0, 8.0), load_N: float = 0.0,
                      motor_torque_Nm: float = 0.0) -> list[dict]:
    """The same screw at several leads: the speed vs thrust vs back-drive choice side by side."""
    return screw.compare(thread, tuple(leads), load_N=load_N, motor_torque_Nm=motor_torque_Nm)


@tool
def lever_actuator(anchor_mm: float, boss_mm: float, angle_from_deg: float, angle_to_deg: float,
                   force_N: float = 0.0, load_torque_Nm: float = 0.0) -> dict:
    """A linear actuator driving a joint through a lever, swept across the joint range: stroke,
    moment arm at every angle, and where the actuator runs out of torque."""
    return mechanism.ram(anchor_mm, boss_mm, angle_from_deg, angle_to_deg, force_N=force_N,
                         load_torque_Nm=load_torque_Nm)


@tool
def arm_workspace(link1_mm: float, link2_mm: float, joint1_deg: float = 95.0, joint2_deg: float = 120.0,
                  tool_mm: float = 0.0) -> dict:
    """Reachable envelope of a two-link planar arm from link lengths and joint limits."""
    return mechanism.workspace(link1_mm, link2_mm, joint1_deg=joint1_deg, joint2_deg=joint2_deg, tool_mm=tool_mm)


@tool
def tipover(track_mm: float, wheelbase_mm: float, cog_height_mm: float, base_kg: float, arm_kg: float,
            payload_kg: float = 0.0, reach_mm: float = 0.0, counterweight_kg: float = 0.0) -> dict:
    """Stability margin around the full slew circle for a base carrying an arm at reach."""
    return mechanism.tipover(track_mm=track_mm, wheelbase_mm=wheelbase_mm, cog_height_mm=cog_height_mm,
                             base_kg=base_kg, arm_kg=arm_kg, payload_kg=payload_kg, reach_mm=reach_mm,
                             counterweight_kg=counterweight_kg)


@tool
def drive_sizing(mass_kg: float, wheel_dia_mm: float, motor_stall_Nm: float = 0.0, gear_ratio: float = 1.0,
                 driven_wheels: int = 4, grade_deg: float = 10.0, crr: float = 0.03, mu: float = 0.9) -> dict:
    """Wheel torque to roll, climb and accelerate, and whether traction or the motor gives out first."""
    return mechanism.drive(mass_kg=mass_kg, wheel_dia_mm=wheel_dia_mm, motor_stall_Nm=motor_stall_Nm,
                           gear_ratio=gear_ratio, driven_wheels=driven_wheels, grade_deg=grade_deg, crr=crr, mu=mu)


@tool
def gear_mesh(module_mm: float, teeth_a: int, teeth_b: int, internal: bool = False,
              placed_centre_mm: float | None = None) -> dict:
    """Correct centre distance for two gears, and whether a placed distance actually meshes."""
    return mechanism.gear_mesh(module_mm, teeth_a, teeth_b, internal=internal, placed_centre_mm=placed_centre_mm)


@tool
def planetary_design(module: float, sun: int, planet: int, n_planets: int = 3, face_mm: float = 8.0) -> dict:
    """Every number a printable planetary gearset needs, or a refusal naming the rule it breaks
    (assembly condition, planet clearance, minimum teeth)."""
    return planetary.design(module, sun, planet, n_planets, face_mm=face_mm)


@tool
def planetary_suggest(module: float = 1.5, n_planets: int = 3, want_ratio: float = 4.0,
                      max_ring_od_mm: float = 200.0) -> list[dict]:
    """Valid planetary gearsets closest to a target ratio."""
    return planetary.suggest(module, n_planets, want_ratio, max_ring_od_mm)


# ── power and mass ───────────────────────────────────────────────────────────────────────────────
@tool
def power_check(components: list[dict], layout: dict | None = None) -> dict:
    """Does the supply match the parts on it. components: [{name, category, voltage_nom_v,
    current_cont_a, qty}], layout: {cells} or {bus_v}. Flags starved and over-volted parts and parts
    whose rating is unknown."""
    return power.check(components, layout)


@tool
def mass_aggregate(items: list[dict]) -> dict:
    """Total mass, centre of gravity and inertia about it. items: [{name, mass_g, x_mm, y_mm, z_mm}]."""
    return massprops.aggregate(items)


@tool
def battery_mass(capacity_mah: float, v_nom: float, wh_per_kg: float = 150.0) -> dict:
    """Pack mass estimated from its energy."""
    return {"ok": True, "mass_g": massprops.battery_mass_g(capacity_mah, v_nom, wh_per_kg)}


# ── printing ─────────────────────────────────────────────────────────────────────────────────────
@tool
def print_hole(nominal_mm: float, material: str = "pla", axis: str = "z") -> dict:
    """The diameter to MODEL so a printed hole comes out at nominal. Uses this machine's measured
    shrink when calibrate_record_hole has data, else a stated assumption."""
    return dfm.hole(nominal_mm, material=material, axis=axis)


@tool
def print_shaft(nominal_mm: float, material: str = "pla", clearance_mm: float = 0.0) -> dict:
    """The diameter to MODEL so a printed pin or spigot comes out at nominal."""
    return dfm.shaft(nominal_mm, material=material, clearance_mm=clearance_mm)


@tool
def print_insert_boss(screw_size: str = "M3", material: str = "pla") -> dict:
    """Hole and boss for a heat-set brass insert (M2 to M8)."""
    return dfm.insert_boss(screw_size, material=material)


@tool
def print_wall(target_mm: float, nozzle_mm: float = 0.4) -> dict:
    """Nearest wall thickness that is a whole number of extrusion lines."""
    return dfm.wall(target_mm, nozzle_mm)


@tool
def calibrate_plan(material: str) -> dict:
    """What coupons to print and measure so the printing and material tools use this machine's
    numbers instead of datasheet ones."""
    return calibrate.plan(material)


@tool
def calibrate_record_hole(material: str, nominal_mm: float, measured_mm: float, printer: str = "") -> dict:
    """Record one measured bore from a printed coupon. Stored in ~/.mechcalc (MECHCALC_DATA_DIR)."""
    return calibrate.record_hole(material, nominal_mm, measured_mm, printer)


# ── structure ────────────────────────────────────────────────────────────────────────────────────
@tool
def fea_beam(length: float, width: float, height: float, load: float, material: str = "aluminum",
             support: str = "cantilever_tip") -> dict:
    """Rectangular beam, point load: max stress, deflection, safety factor. mm, N, MPa."""
    return fea.beam(length, width, height, load, material, support)


@tool
def fea_column(length: float, width: float, height: float, material: str = "aluminum", end: str = "pinned") -> dict:
    """Euler buckling vs yield for a rectangular column."""
    return fea.column(length, width, height, material, end)


@tool
def fea_bolt(torque_nm: float, diameter_mm: float, k: float = 0.2) -> dict:
    """Bolt preload and tensile stress from tightening torque."""
    return fea.bolt(torque_nm, diameter_mm, k)


@tool
def fea_thermal(length_mm: float, delta_t: float, material: str = "aluminum", constrained: bool = False) -> dict:
    """Thermal expansion, and stress if fully restrained."""
    return fea.thermal(length_mm, delta_t, material, constrained)


# ── wearables ────────────────────────────────────────────────────────────────────────────────────
@tool
def wearable_pressure(force_n: float, pad_area_cm2: float, hours: float = 8.0, site: str = "cuff") -> dict:
    """Skin interface pressure from a strap or cuff against wear-time limits."""
    return wearable.interface_pressure(force_n, pad_area_cm2, hours, site=site)


@tool
def wearable_capacity(pad_area_cm2: float, site: str = "cuff", sustained: bool = True) -> dict:
    """Most force a pad of this area can carry into skin."""
    return wearable.load_capacity(pad_area_cm2, site, sustained)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
