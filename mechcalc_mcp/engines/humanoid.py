"""humanoid.py — bipedal humanoid statics (first-cut). Worst-case leg joint torque in single-support,
static stability margin (CoG over the foot polygon), holding power + runtime, payload, and mass. A
static-balance model; dynamic walking (ZMP) is a later upgrade. Metric keys match the humanoid archetype.
"""
from __future__ import annotations

from ._common import G, cfg, keybar, render_2x2

DEFAULTS = {
    "height_mm": 400.0, "mass_g": 2500.0, "dof": 16, "thigh_mm": 130.0, "shin_mm": 130.0,
    "foot_len_mm": 90.0, "joint_stall_torque_Nm": 3.0, "servo_mass_g": 60.0, "payload_g": 200.0,
    "cells": 3, "batt_capacity_mah": 2200.0, "joint_hold_power_W": 1.2, "safety_factor": 1.5,
}


def analyze(design: dict | None = None) -> dict:
    d = cfg(DEFAULTS, design)
    warn: list[str] = []
    mass_kg = (d["mass_g"] + d["payload_g"]) / 1000.0
    weight_N = mass_kg * G
    # single-support: the stance hip/ankle carries the whole body at a moment arm ~ half a step
    step_arm_m = d["thigh_mm"] / 2000.0
    req_torque = weight_N * step_arm_m
    margin = d["joint_stall_torque_Nm"] / req_torque if req_torque > 0 else float("inf")
    # static stability: foot polygon vs CoG height (taller = higher CoG = less stable)
    stability = max(0.0, min(1.0, (d["foot_len_mm"] / d["height_mm"]) * 3.0 - d["payload_g"] / max(1.0, d["mass_g"] * 2)))
    v_pack = d["cells"] * 3.7
    power_W = d["dof"] * d["joint_hold_power_W"]
    runtime_min = (d["batt_capacity_mah"] / 1000.0 * v_pack) / power_W * 60.0 if power_W > 0 else 0.0

    if margin < d["safety_factor"]:
        warn.append(f"leg joint margin {margin:.2f} < SF {d['safety_factor']:g} — it can't hold single-support; "
                    f"stronger hip/knee actuators, lighter build, or shorter legs (needs {req_torque:.2f} N·m)")
    if stability < 0.3:
        warn.append(f"low static stability ({stability:.2f}) — bigger feet, lower CoG, or keep it in double-support")
    if runtime_min < 10:
        warn.append(f"runtime {runtime_min:.0f} min — more battery or lower-power (torque-holding) joints")
    return {
        "metrics": {
            "payload_g": round(d["payload_g"], 0),
            "joint_torque_margin": round(margin, 2),
            "stability_margin": round(stability, 2),
            "power_W": round(power_W, 1),
            "mass_kg": round(mass_kg, 2),
            "runtime_min": round(runtime_min, 0),
            "req_joint_torque_Nm": round(req_torque, 2),
        },
        "warnings": warn,
        "_curves": {"d": d, "weight_N": weight_N, "stall": d["joint_stall_torque_Nm"], "power": power_W},
    }


def plot(design: dict | None, out_png) -> dict:
    import numpy as np
    r = analyze(design)
    c = r["_curves"]
    m = r["metrics"]
    d = c["d"]

    def draw(ax):
        arms = np.linspace(20, d["thigh_mm"], 100)
        tq = c["weight_N"] * arms / 2000.0
        ax[0, 0].plot(arms, tq, lw=2); ax[0, 0].axhline(c["stall"], ls="--", c="g", label=f"joint {c['stall']:.1f} N·m")
        ax[0, 0].set_xlabel("step moment arm (mm)"); ax[0, 0].set_ylabel("hip torque (N·m)")
        ax[0, 0].set_title("Joint torque (single-support)"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
        h = np.linspace(200, 800, 100)
        stab = np.clip(d["foot_len_mm"] / h * 3.0 - d["payload_g"] / max(1, d["mass_g"] * 2), 0, 1)
        ax[0, 1].plot(h, stab, lw=2, color="#14b8a6"); ax[0, 1].axvline(d["height_mm"], ls=":", c="#888")
        ax[0, 1].axhline(0.3, ls="--", c="#d33", label="min stable")
        ax[0, 1].set_xlabel("height (mm)"); ax[0, 1].set_ylabel("stability margin")
        ax[0, 1].set_title("Stability vs height"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3)
        ax[1, 0].bar(["torque SF", "stability"], [min(m["joint_torque_margin"], 3), m["stability_margin"]], color=["#8b5cf6", "#14b8a6"])
        ax[1, 0].axhline(1, ls="--", c="#999"); ax[1, 0].set_title("Margins"); ax[1, 0].grid(alpha=.3)
        keybar(ax[1, 1], [("payload g", m["payload_g"]), ("joint SF", m["joint_torque_margin"]),
                          ("stability", m["stability_margin"]), ("power W", m["power_W"]),
                          ("mass kg", m["mass_kg"])])
    render_2x2(out_png, draw)
    return {"metrics": m, "warnings": r["warnings"]}


if __name__ == "__main__":
    print("humanoid:", analyze()["metrics"])
