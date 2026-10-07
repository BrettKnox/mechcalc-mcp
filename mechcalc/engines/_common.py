"""_common.py — shared primitives for the deterministic domain engines (arm/cnc/drone/legged/humanoid/
linear/balance). First-principles, 0 LLM tokens. Each engine mirrors analysis.py: DEFAULTS +
analyze() (pure math) + plot() (lazy numpy/matplotlib, 2x2 panel). Units: mm, N, g, MPa in; SI-ish out.
"""
from __future__ import annotations

import math

G = 9.81
# Views over the one material table (matprops.py). These stay as dicts so existing importers keep
# working, but nothing should .get(..., default) off them: a missing key used to substitute a different
# material silently (E_of("brass") returned PLA's 3500, a 33x error). Use the accessors below.
from .. import matprops as _mp                                                          # noqa: E402

E_MPA = _mp.table("E_mpa")
DENSITY = _mp.table("density_kgm3")


def cfg(defaults: dict, design: dict | None) -> dict:
    """Coerce+filter a design dict to the engine's DEFAULTS (types + whitelist), like analysis._cfg."""
    d = dict(defaults)
    for k, v in (design or {}).items():
        if k in d:
            if defaults[k] is None:
                # A None default means "no type to coerce to", not "reject everything". Coercing
                # through type(None) raises, and the except below then swallows the override —
                # so the key would silently ignore every value ever passed for it.
                d[k] = v
                continue
            try:
                d[k] = type(defaults[k])(v)
            except (TypeError, ValueError):
                pass
    return d


def E_of(material: str) -> float:
    """Young's modulus, MPa. Raises UnknownMaterial rather than quietly returning PLA's."""
    return float(_mp.get(material).E_mpa)


def rho_of(material: str) -> float:
    """Density, kg/m^3. Raises rather than quietly returning PLA's."""
    return float(_mp.get(material).density_kgm3)


def yield_of(material: str) -> float:
    """The stress to design against, knockdown applied. Brittle materials use uts with a factor."""
    return float(_mp.get(material).allowable_mpa)


def z_ratio_of(material: str) -> float:
    """Strength across print layers relative to along them (1.0 for stock metal)."""
    return float(_mp.get(material).z_strength_ratio)


def tube_I_mm4(od_mm: float, wall_mm: float) -> float:
    """Second moment of area of a round tube (mm^4). Solid if wall >= od/2."""
    id_mm = max(0.0, od_mm - 2 * wall_mm)
    return math.pi / 64.0 * (od_mm ** 4 - id_mm ** 4)


def tube_area_mm2(od_mm: float, wall_mm: float) -> float:
    id_mm = max(0.0, od_mm - 2 * wall_mm)
    return math.pi / 4.0 * (od_mm ** 2 - id_mm ** 2)


def cantilever_defl_mm(load_N: float, L_mm: float, E_mpa: float, I_mm4: float) -> float:
    """Tip deflection of a cantilever, point load at the tip: P*L^3/(3*E*I)."""
    return load_N * L_mm ** 3 / (3.0 * E_mpa * I_mm4) if I_mm4 > 0 else float("inf")


def simply_supported_defl_mm(load_N: float, L_mm: float, E_mpa: float, I_mm4: float) -> float:
    """Mid-span deflection, central point load: P*L^3/(48*E*I)."""
    return load_N * L_mm ** 3 / (48.0 * E_mpa * I_mm4) if I_mm4 > 0 else float("inf")


def keybar(ax, pairs):
    """Render a horizontal bar of (label, value) key metrics on a matplotlib axis."""
    labels = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    ax.barh(labels, vals, color="#0ea5e9")
    ax.set_title("Key metrics")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:g}", va="center", fontsize=9)


def render_2x2(out_png, draw):
    """Lazy-import matplotlib, make a 2x2 figure, call draw(axes), save at dpi 120. Returns nothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    draw(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ── stress, margin and first mode ─────────────────────────────────────────────
# The engines computed deflection but never stress, so no design ever reported whether it would YIELD —
# only whether it would bend. A part can pass a deflection limit and still snap. All closed-form,
# math-only and microseconds, so the optimiser can call them inside its fitness function.

def bending_stress_mpa(moment_Nmm: float, c_mm: float, I_mm4: float) -> float:
    """Bending stress sigma = M*c/I. Inputs N*mm, mm, mm^4 -> MPa (N/mm^2)."""
    if I_mm4 <= 0:
        return 0.0
    return round(moment_Nmm * c_mm / I_mm4, 2)


def cantilever_stress_mpa(load_N: float, L_mm: float, c_mm: float, I_mm4: float) -> float:
    """Peak stress at the root of a tip-loaded cantilever (M = P*L)."""
    return bending_stress_mpa(load_N * L_mm, c_mm, I_mm4)


def safety_factor(sigma_mpa: float, material: str, across_layers: bool = False) -> float:
    """Allowable / applied. `across_layers` applies the printed z-strength ratio, which is what makes
    orienting a printed part along its load path matter: PLA is only ~50% as strong across layers."""
    if sigma_mpa <= 0:
        return float("inf")
    allow = yield_of(material)
    if across_layers:
        allow *= z_ratio_of(material)
    return round(allow / sigma_mpa, 2)


def first_mode_hz(E_mpa: float, I_mm4: float, mass_kg: float, L_mm: float) -> float:
    """First bending mode of a tip-loaded cantilever (Rayleigh, lumped mass): f = 0.560*sqrt(k/m),
    k = 3EI/L^3. A structure whose first mode sits near the drive/step frequency shakes itself apart,
    which nothing in the app could previously see."""
    if mass_kg <= 0 or L_mm <= 0 or I_mm4 <= 0:
        return 0.0
    k_N_per_m = 3.0 * E_mpa * I_mm4 / (L_mm ** 3) * 1000.0     # N/mm -> N/m
    return round(0.560 * math.sqrt(k_N_per_m / mass_kg), 1)


def apply_derate(design: dict, env_rules: dict | None, material_keys=()) -> dict:
    """Scale a design by the environment's derates. Returns a NEW dict; never mutates the caller's.

    `environment.rules_for()` has always computed these multipliers and nothing ever read them, so the
    same configuration returned byte-identical numbers at 20 C and at 55 C in direct sun. That is the
    "ideal-vacuum design" failure: heat softens polymers, cold robs battery capacity, altitude thins the
    air, and none of it reached a single number.
    """
    d = dict(design or {})
    dr = ((env_rules or {}).get("derate") or {})
    if not dr:
        return d
    batt = dr.get("battery_capacity")
    if batt and d.get("batt_capacity_mah"):
        d["batt_capacity_mah"] = round(float(d["batt_capacity_mah"]) * float(batt), 1)
    thermal = dr.get("motor_thermal_headroom")
    if thermal:
        for k in ("motor_cont_current_a", "motor_stall_torque_Nm", "servo_stall_torque_Nm",
                  "joint_stall_torque_Nm", "stepper_torque_Nm", "motor_torque_Nm"):
            if d.get(k):
                d[k] = round(float(d[k]) * float(thermal), 4)
    rho = dr.get("air_density")
    if rho:
        for k in ("prop_ct", "thrust_per_watt_N"):
            if d.get(k):
                d[k] = round(float(d[k]) * float(rho), 5)
    return d


def derated_E(material: str, env_rules: dict | None) -> float:
    """Young's modulus after the environment's polymer-stiffness derate. Metals are unaffected."""
    E = E_of(material)
    f = ((env_rules or {}).get("derate") or {}).get("polymer_E")
    from .. import matprops as _m
    if f and _m.get(material).printable:
        return round(E * float(f), 1)
    return E
