"""analysis.py — first-principles vehicle multiphysics, deterministic (numpy only for plots, no model calls).

Given a design (mass, aero, motor, gearing, battery, wheels), it computes the coupled longitudinal
behaviour a real RC/EV designer checks: tractive vs resistive force across speed, top speed, 0-to-v
acceleration (traction- and power-limited), electrical draw / runtime / battery C-rate headroom, and
motor ohmic heating vs its continuous rating. Assumptions are explicit; this is a 1D lumped model
(quasi-static motor curve + point-mass dynamics), not CFD/FEA — those escalate to compute/GPU when the
ROI justifies it (see refine.py). Units SI internally; convenience km/h at the edges.
"""
from __future__ import annotations

import math

G = 9.81
RHO_AIR = 1.225
MU_TIRE = 1.0          # rubber-on-pavement traction coefficient (launch grip limit)

# sensible 1/10-scale RC defaults; every field is overridable per design.
# NOTE: numeric defaults must be FLOAT literals — _cfg coerces with type(DEFAULTS[k])(v), so an int
# default would silently truncate a real value (which is why `cells` floors a fractional input).
DEFAULTS = {
    "mass_kg": 1.6, "wheel_d_mm": 64.0, "cd": 0.40, "frontal_area_cm2": 105.0,
    "crr": 0.015, "drivetrain_eff": 0.85, "gear_ratio": 4.0,
    "motor_kv": 3500.0, "motor_max_current_a": 45.0, "motor_cont_current_a": 25.0,
    "motor_resistance_ohm": 0.02, "cells": 3,
    "batt_capacity_mah": 5000.0, "batt_c_rating": 50.0,
    # ── layout / weight transfer: real traction instead of "all the mass is on the driven axle" ──
    "wheelbase_mm": 257.0,     # axle centres (1/10 touring default)
    "track_mm": 190.0,         # left-right wheel centres (lateral tip check)
    "cog_height_mm": 45.0,     # CoG above ground — the transfer lever arm
    "cog_x_mm": 0.0,           # CoG fore(+)/aft(-) of mid-wheelbase
    "drive": "awd",            # awd | rear | front
    "mu_tire": 1.0,            # tyre-surface friction coefficient
}


def _cfg(design: dict | None) -> dict:
    d = dict(DEFAULTS)
    for k, v in (design or {}).items():
        if k in d:
            try:
                d[k] = type(DEFAULTS[k])(v)
            except (TypeError, ValueError):
                pass
    return d


def analyze(design: dict | None = None) -> dict:
    """The full report: metrics dict + a list of engineering warnings (things that don't close)."""
    d = _cfg(design)
    warn: list[str] = []
    v_pack = d["cells"] * 3.7                                   # nominal LiPo pack voltage
    kt = 9.5493 / d["motor_kv"]                                 # N*m/A  (= 60/(2*pi*Kv))
    rpm_noload = d["motor_kv"] * v_pack
    t_stall = kt * d["motor_max_current_a"]                     # N*m at max current
    r_wheel = d["wheel_d_mm"] / 2000.0                          # m
    area = d["frontal_area_cm2"] / 1e4                          # m^2
    eff, gr, m = d["drivetrain_eff"], d["gear_ratio"], d["mass_kg"]
    f_roll = d["crr"] * m * G

    # ── launch traction from the actual weight distribution ──
    # The driven axle only carries its share of the weight, and accelerating transfers load rearward
    # (which is why a rear-drive car hooks up and a front-drive one spins). Closed form, no iteration:
    #   rear:  F = mu*m*g*w_r0 / (1 - mu*h/L)      front: F = mu*m*g*w_f0 / (1 + mu*h/L)
    # capped by the wheelie limit, where the front normal load reaches zero.
    mu = max(0.0, d["mu_tire"])
    L = max(1e-6, d["wheelbase_mm"] / 1000.0)                   # m
    h = max(0.0, d["cog_height_mm"] / 1000.0)                   # m
    w_r0 = min(1.0, max(0.0, 0.5 - (d["cog_x_mm"] / 1000.0) / L))   # static rear weight fraction
    w_f0 = 1.0 - w_r0
    drive = str(d["drive"]).lower()
    if drive in ("rear", "rwd", "r"):
        denom = 1.0 - mu * h / L
        f_slip = (mu * m * G * w_r0 / denom) if denom > 0 else float("inf")
    elif drive in ("front", "fwd", "f"):
        f_slip = mu * m * G * w_f0 / (1.0 + mu * h / L)
    else:                                                       # awd/4wd: every wheel drives
        f_slip = mu * m * G
    f_wheelie = (m * G * w_f0 * L / h) if h > 0 else float("inf")
    f_grip = min(f_slip, f_wheelie)                             # launch traction ceiling
    if f_wheelie < f_slip:
        warn.append(f"wheelie-limited at launch (front lifts at {f_wheelie:.0f} N) — "
                    "lower the CoG or move mass forward")
    if drive in ("front", "fwd", "f") and w_f0 < 0.45:
        warn.append(f"front-wheel drive with only {w_f0 * 100:.0f}% static front weight — poor launch grip")

    def motor_rpm(v):  # motor rpm at road speed v (m/s)
        return 60.0 * v * gr / (2 * math.pi * r_wheel)

    def f_trac(v):     # available tractive force, motor torque falls linearly to 0 at no-load rpm
        t = t_stall * max(0.0, 1.0 - motor_rpm(v) / rpm_noload)
        return t * gr * eff / r_wheel

    def f_resist(v):
        return 0.5 * RHO_AIR * d["cd"] * area * v * v + f_roll

    # top speed: highest v where available thrust still beats resistance
    vs = [i * 0.05 for i in range(1, 1400)]                     # 0..70 m/s
    v_top = 0.0
    for v in vs:
        if min(f_trac(v), f_grip) - f_resist(v) > 0:
            v_top = v
        else:
            break

    # 0 -> target acceleration by integrating dt = dv / a(v)
    def accel_time(target_ms):
        t, v, dv = 0.0, 0.0, 0.02
        while v < target_ms:
            a = (min(f_trac(v), f_grip) - f_resist(v)) / m
            if a <= 1e-3:
                return None                                    # can't reach that speed
            t += dv / a
            v += dv
            if t > 60:
                return None
        return t

    t_30 = accel_time(30 / 3.6)
    t_50 = accel_time(50 / 3.6)
    a0 = (min(f_trac(0.0), f_grip) - f_resist(0.0)) / m        # launch accel (m/s^2)
    launch_grip_limited = f_trac(0.0) > f_grip

    # electrical + thermal
    i_peak = d["motor_max_current_a"]
    i_batt_max = d["batt_c_rating"] * d["batt_capacity_mah"] / 1000.0
    p_peak = v_pack * i_peak
    runtime_cont_min = (d["batt_capacity_mah"] / 1000.0) / d["motor_cont_current_a"] * 60.0
    p_loss_cont = d["motor_cont_current_a"] ** 2 * d["motor_resistance_ohm"]
    p_loss_peak = i_peak ** 2 * d["motor_resistance_ohm"]

    if i_peak > i_batt_max:
        warn.append(f"battery can't source peak: {i_peak:.0f} A demanded > {i_batt_max:.0f} A "
                    f"({d['batt_c_rating']:.0f}C x {d['batt_capacity_mah']/1000:.1f} Ah) — raise C-rate or capacity")
    if launch_grip_limited:
        warn.append(f"launch is traction-limited ({f_trac(0.0):.0f} N thrust > {f_grip:.0f} N grip) — "
                    f"power past the tires' limit is wasted; wider/stickier tires or less initial torque")
    if t_30 is None:
        warn.append("cannot reach 30 km/h — under-geared or under-powered for the drag+rolling load")
    if p_loss_peak > 5 * p_loss_cont:
        warn.append(f"peak motor heating {p_loss_peak:.0f} W is high vs continuous {p_loss_cont:.0f} W — "
                    f"limit peak duration or improve cooling")

    return {
        "metrics": {
            "top_speed_kmh": round(v_top * 3.6, 1),
            "accel_0_30kmh_s": round(t_30, 2) if t_30 else None,
            "accel_0_50kmh_s": round(t_50, 2) if t_50 else None,
            "launch_accel_g": round(a0 / G, 2),
            "peak_power_W": round(p_peak, 0),
            "peak_current_A": round(i_peak, 1),
            "batt_source_max_A": round(i_batt_max, 0),
            "runtime_cont_min": round(runtime_cont_min, 1),
            "motor_heat_cont_W": round(p_loss_cont, 1),
            "motor_heat_peak_W": round(p_loss_peak, 1),
            "gear_ratio": round(gr, 2), "mass_kg": round(m, 2),
            "drag_area_CdA": round(d["cd"] * area, 4),
            # layout-derived (additive — objective_score is unaffected)
            "weight_dist_rear_pct": round(w_r0 * 100.0, 1),
            "grip_limit_N": round(f_grip, 1),
            "traction_accel_g": round(f_grip / m / G, 2),
            "tip_over_lateral_g": round((d["track_mm"] / 2.0) / d["cog_height_mm"], 2) if d["cog_height_mm"] > 0 else None,
        },
        "warnings": warn,
        "_curves": {"f_trac": f_trac, "f_resist": f_resist, "v_top": v_top, "grip": f_grip},
    }


def plot(design: dict | None, out_png) -> dict:
    """Run analyze() and render a 2x2 engineering panel to out_png. Returns metrics+warnings (no funcs)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = analyze(design)
    c = r["_curves"]
    v = np.linspace(0.01, max(c["v_top"] * 1.3, 5), 200)
    ft = np.array([c["f_trac"](x) for x in v])
    fr = np.array([c["f_resist"](x) for x in v])
    grip = c["grip"]
    m = r["metrics"]
    kmh = v * 3.6
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].plot(kmh, np.minimum(ft, grip), label="tractive (grip-capped)", lw=2)
    ax[0, 0].plot(kmh, fr, label="resistance (drag+roll)", lw=2, color="#d33")
    ax[0, 0].axvline(m["top_speed_kmh"], ls=":", c="g", label=f"top speed {m['top_speed_kmh']} km/h")
    ax[0, 0].set_xlabel("speed (km/h)"); ax[0, 0].set_ylabel("force (N)"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
    ax[0, 0].set_title("Force balance -> top speed")
    accel = (np.minimum(ft, grip) - fr) / m["mass_kg"] / G
    ax[0, 1].plot(kmh, accel, lw=2, color="#8b5cf6"); ax[0, 1].axhline(0, ls="--", c="#999")
    ax[0, 1].set_xlabel("speed (km/h)"); ax[0, 1].set_ylabel("acceleration (g)"); ax[0, 1].grid(alpha=.3)
    ax[0, 1].set_title(f"Acceleration  (0-30 km/h: {m['accel_0_30kmh_s']} s)")
    drag = 0.5 * RHO_AIR * design_cd(design) * design_area(design) * v ** 2
    ax[1, 0].plot(kmh, drag, label="aero drag", lw=2); ax[1, 0].axhline(design_roll(design), ls="--", c="#f59e0b", label="rolling")
    ax[1, 0].set_xlabel("speed (km/h)"); ax[1, 0].set_ylabel("force (N)"); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3)
    ax[1, 0].set_title("Resistance breakdown")
    labels = ["top km/h", "0-30 s", "launch g", "peak kW", "run min"]
    vals = [m["top_speed_kmh"], m["accel_0_30kmh_s"] or 0, m["launch_accel_g"], m["peak_power_W"] / 1000, m["runtime_cont_min"]]
    ax[1, 1].barh(labels, vals, color="#0ea5e9"); ax[1, 1].set_title("Key metrics")
    for i, val in enumerate(vals):
        ax[1, 1].text(val, i, f" {val:g}", va="center", fontsize=9)
    plt.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return {"metrics": r["metrics"], "warnings": r["warnings"]}


# tiny helpers so plot() doesn't re-open the cfg
def design_cd(d): return _cfg(d)["cd"]
def design_area(d): return _cfg(d)["frontal_area_cm2"] / 1e4
def design_roll(d):
    c = _cfg(d); return c["crr"] * c["mass_kg"] * G


if __name__ == "__main__":   # self-check: physics is sane on RC defaults
    r = analyze()
    m = r["metrics"]
    assert m["top_speed_kmh"] > 5, m
    assert m["accel_0_30kmh_s"] and m["accel_0_30kmh_s"] > 0, m
    assert m["peak_power_W"] > 0 and m["runtime_cont_min"] > 0, m
    print("analysis OK:", m)
    for w in r["warnings"]:
        print("  warn:", w)
