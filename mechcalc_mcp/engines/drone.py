"""drone.py — multirotor performance. Static prop thrust (thrust-coefficient model), all-up weight,
thrust-to-weight, hover current + flight time, and payload headroom. Covers the mini FPV drone and the
micro swarm. Metric keys match the quadcopter archetype objectives.
"""
from __future__ import annotations

from ._common import G, cfg, keybar, render_2x2

RHO = 1.225
IN_M = 0.0254

DEFAULTS = {
    "n_motors": 4, "motor_kv": 2300.0, "motor_max_current_a": 25.0, "motor_mass_g": 32.0,
    "prop_diameter_in": 5.0, "prop_pitch_in": 4.5, "prop_ct": 0.11, "cells": 4,
    "batt_capacity_mah": 1500.0, "batt_c_rating": 75.0, "frame_mass_g": 250.0,
    "esc_mass_g": 30.0, "payload_g": 0.0, "thrust_per_watt_N": 0.0785,   # ~8 g/W props
}


def _max_thrust_per_motor_N(d, v_pack):
    n_rev_s = d["motor_kv"] * v_pack * 0.82 / 60.0        # loaded full-throttle rev/s
    D = d["prop_diameter_in"] * IN_M
    return d["prop_ct"] * RHO * n_rev_s ** 2 * D ** 4     # T = Ct*rho*n^2*D^4


def analyze(design: dict | None = None) -> dict:
    d = cfg(DEFAULTS, design)
    warn: list[str] = []
    v_pack = d["cells"] * 3.7
    batt_mass = d["batt_capacity_mah"] / 1000.0 * v_pack / 150.0 * 1000.0    # g, ~150 Wh/kg LiPo
    mass_g = d["frame_mass_g"] + d["n_motors"] * d["motor_mass_g"] + d["n_motors"] * d["esc_mass_g"] + batt_mass + d["payload_g"]
    mass_kg = mass_g / 1000.0
    t_motor = _max_thrust_per_motor_N(d, v_pack)
    total_thrust = d["n_motors"] * t_motor
    twr = total_thrust / (mass_kg * G) if mass_kg > 0 else 0.0
    hover_thrust_N = mass_kg * G
    hover_power_W = hover_thrust_N / d["thrust_per_watt_N"]
    hover_current_A = hover_power_W / v_pack
    flight_time = (d["batt_capacity_mah"] / 1000.0 * 0.8) / hover_current_A * 60.0 if hover_current_A > 0 else 0.0
    base_kg = (mass_g - d["payload_g"]) / 1000.0
    payload_cap_g = max(0.0, (total_thrust / (1.5 * G) - base_kg) * 1000.0)   # keep T/W>=1.5 with payload

    i_batt_max = d["batt_c_rating"] * d["batt_capacity_mah"] / 1000.0
    i_peak = d["n_motors"] * d["motor_max_current_a"]
    if twr < 2.0:
        warn.append(f"thrust-to-weight {twr:.2f} < 2 — too sluggish to control; lighter build, bigger prop/Kv, or more cells")
    if i_peak > i_batt_max:
        warn.append(f"battery can't source full throttle: {i_peak:.0f} A > {i_batt_max:.0f} A ({d['batt_c_rating']:.0f}C) — higher C or capacity")
    if flight_time < 4:
        warn.append(f"flight time {flight_time:.1f} min is short — bigger battery (costs weight → T/W) or more efficient props")
    return {
        "metrics": {
            "thrust_to_weight": round(twr, 2),
            "flight_time_min": round(flight_time, 1),
            "payload_g": round(payload_cap_g, 0),
            "mass_kg": round(mass_kg, 3),
            "max_thrust_g": round(total_thrust / G * 1000, 0),
            "hover_current_A": round(hover_current_A, 1),
        },
        "warnings": warn,
        "_curves": {"d": d, "v_pack": v_pack, "mass_kg": mass_kg, "t_motor": t_motor,
                    "twr": twr, "hover_A": hover_current_A, "base_kg": base_kg, "batt_mass": batt_mass},
    }


def plot(design: dict | None, out_png) -> dict:
    import numpy as np
    r = analyze(design)
    c = r["_curves"]
    m = r["metrics"]
    d = c["d"]

    def draw(ax):
        kvs = np.linspace(1000, 4000, 100)
        n = kvs * c["v_pack"] * 0.82 / 60.0
        D = d["prop_diameter_in"] * IN_M
        tw = d["n_motors"] * d["prop_ct"] * RHO * n ** 2 * D ** 4 / (c["mass_kg"] * G)
        ax[0, 0].plot(kvs, tw, lw=2)
        ax[0, 0].axhline(2.0, ls="--", c="#d33", label="min T/W = 2"); ax[0, 0].axvline(d["motor_kv"], ls=":", c="g")
        ax[0, 0].set_xlabel("motor Kv"); ax[0, 0].set_ylabel("thrust-to-weight")
        ax[0, 0].set_title("T/W vs Kv"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
        caps = np.linspace(500, 3000, 100)
        bm = caps / 1000.0 * c["v_pack"] / 150.0
        auw = (c["mass_kg"] - c["batt_mass"] / 1000.0) + bm
        hp = auw * G / d["thrust_per_watt_N"]; hi = hp / c["v_pack"]
        ft = (caps / 1000.0 * 0.8) / hi * 60.0
        ax[0, 1].plot(caps, ft, lw=2, color="#0ea5e9"); ax[0, 1].axvline(d["batt_capacity_mah"], ls=":", c="g")
        ax[0, 1].set_xlabel("battery (mAh)"); ax[0, 1].set_ylabel("flight time (min)")
        ax[0, 1].set_title("Flight time vs battery"); ax[0, 1].grid(alpha=.3)
        ax[1, 0].bar(["frame", "motors+esc", "battery", "payload"],
                     [d["frame_mass_g"], d["n_motors"] * (d["motor_mass_g"] + d["esc_mass_g"]), c["batt_mass"], d["payload_g"]],
                     color="#8b5cf6")
        ax[1, 0].set_ylabel("mass (g)"); ax[1, 0].set_title("Mass breakdown"); ax[1, 0].grid(alpha=.3)
        keybar(ax[1, 1], [("T/W", m["thrust_to_weight"]), ("flight min", m["flight_time_min"]),
                          ("payload g", m["payload_g"]), ("AUW g", round(m["mass_kg"] * 1000)),
                          ("hover A", m["hover_current_A"])])
    render_2x2(out_png, draw)
    return {"metrics": m, "warnings": r["warnings"]}


if __name__ == "__main__":
    print("drone:", analyze()["metrics"])
