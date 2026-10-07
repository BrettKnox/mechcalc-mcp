"""linear.py — linear actuator / mechanism (leadscrew, belt, or servo-lever). Output force, speed,
stroke, holding force (backdrive), and force margin vs the load — the physics behind a remote door
closer, a linear slide, or a gripper. Metric keys match the door_closer / linear_mech archetype.
"""
from __future__ import annotations

import math

from ._common import cfg, keybar, render_2x2

DEFAULTS = {
    # `leadscrew_pitch_mm` is misnamed and kept for compatibility: it has always been used as the
    # LEAD (travel per turn), not the pitch (thread to thread). For a T8 those differ by the start
    # count -- pitch is 2 mm on every T8 ever made, lead is 2, 4 or 8 -- and mistaking one for the
    # other is a 4x error in force that looks like nothing on a drawing.
    "drive": "leadscrew", "motor_torque_Nm": 0.4, "leadscrew_pitch_mm": 8.0, "efficiency": 0.4,
    "motor_rpm": 200.0, "stroke_mm": 100.0, "load_N": 50.0, "lever_arm_mm": 150.0,
    "reduction": 1.0, "safety_factor": 1.5,
    "screw_thread": "T8", "screw_friction": "steel_brass_greased",
}


def analyze(design: dict | None = None) -> dict:
    d = cfg(DEFAULTS, design)
    warn: list[str] = []
    drive = (d["drive"] or "leadscrew").lower()
    torque = d["motor_torque_Nm"] * d["reduction"]
    if drive == "leadscrew":
        lead_m = d["leadscrew_pitch_mm"] / 1000.0
        # Efficiency and self-locking come from the screw's GEOMETRY, not from a typed-in number.
        # This used to be `holding = force if efficiency <= 0.5`, which decided whether the screw
        # self-locks from an efficiency the designer had assumed -- and the efficiency had been
        # assumed because the screw was believed to self-lock. The lead angle breaks the circle.
        sc = None
        try:
            from .. import screw as _screw
            sc = _screw.analyse(d["screw_thread"], d["leadscrew_pitch_mm"],
                                friction=d["screw_friction"])
        except Exception as e:                             # noqa: BLE001 — an unknown thread is not fatal
            warn.append(f"screw mechanics unavailable ({type(e).__name__}), so efficiency "
                        f"{d['efficiency']:g} is the declared one and self-locking is UNVERIFIED")
        eta = sc["efficiency"] if sc else d["efficiency"]
        force = 2 * math.pi * torque * eta / lead_m if lead_m > 0 else 0.0
        speed = d["motor_rpm"] / 60.0 * d["leadscrew_pitch_mm"]
        holding = force if (sc["self_locking"] if sc else d["efficiency"] <= 0.5) else 0.0
        if sc:
            warn.extend(sc["warnings"])
            if abs(eta - d["efficiency"]) > 0.05:
                warn.append(
                    f"declared efficiency {d['efficiency']:g} but a {sc['lead_mm']:g} mm lead on "
                    f"{sc['thread']} works out at {eta:.2f}. The geometry wins; the declared value "
                    f"was describing a different screw.")
    elif drive == "lever":                                          # servo/motor swinging a lever
        force = torque / (d["lever_arm_mm"] / 1000.0) if d["lever_arm_mm"] > 0 else 0.0
        speed = d["motor_rpm"] / 60.0 * 2 * math.pi * (d["lever_arm_mm"])   # tip speed mm/s
        holding = force
    else:                                                          # belt
        force = 2 * math.pi * torque * d["efficiency"] / (40.0 / 1000.0)
        speed = d["motor_rpm"] / 60.0 * 40.0
        holding = 0.0
    margin = force / d["load_N"] if d["load_N"] > 0 else float("inf")

    if margin < d["safety_factor"]:
        warn.append(f"force margin {margin:.2f} < SF {d['safety_factor']:g} — more motor torque, more reduction, "
                    f"or a finer screw (delivers {force:.0f} N, load {d['load_N']:.0f} N)")
    if speed < 5:
        warn.append(f"actuation is slow ({speed:.0f} mm/s) — coarser screw/higher rpm if it needs to be quick")
    if holding < d["load_N"] and drive != "leadscrew":
        warn.append("won't hold position under load without power — add a brake or a self-locking screw")
    return {
        "metrics": {
            "force_N": round(force, 1),
            "stroke_mm": round(d["stroke_mm"], 0),
            "speed_mms": round(speed, 1),
            "holding_force_N": round(holding, 1),
            "torque_margin": round(margin, 2),
            "required_load_N": round(d["load_N"], 1),
        },
        "warnings": warn,
        "_curves": {"d": d, "force": force, "torque": torque, "drive": drive},
    }


def plot(design: dict | None, out_png) -> dict:
    import numpy as np
    r = analyze(design)
    c = r["_curves"]
    m = r["metrics"]
    d = c["d"]

    def draw(ax):
        pitches = np.linspace(2, 20, 100)
        force = 2 * math.pi * c["torque"] * d["efficiency"] / (pitches / 1000.0)
        ax[0, 0].plot(pitches, force, lw=2); ax[0, 0].axhline(d["load_N"], ls="--", c="#d33", label=f"load {d['load_N']:.0f} N")
        ax[0, 0].axvline(d["leadscrew_pitch_mm"], ls=":", c="g")
        ax[0, 0].set_xlabel("screw pitch (mm)"); ax[0, 0].set_ylabel("output force (N)")
        ax[0, 0].set_title("Force vs pitch"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
        rpms = np.linspace(50, 600, 100)
        speed = rpms / 60.0 * d["leadscrew_pitch_mm"]
        ax[0, 1].plot(rpms, speed, lw=2, color="#0ea5e9"); ax[0, 1].axvline(d["motor_rpm"], ls=":", c="g")
        ax[0, 1].set_xlabel("motor rpm"); ax[0, 1].set_ylabel("speed (mm/s)")
        ax[0, 1].set_title("Speed vs rpm"); ax[0, 1].grid(alpha=.3)
        ax[1, 0].bar(["force", "load"], [m["force_N"], m["required_load_N"]], color=["#8b5cf6", "#d33"])
        ax[1, 0].set_ylabel("N"); ax[1, 0].set_title("Force vs load"); ax[1, 0].grid(alpha=.3)
        keybar(ax[1, 1], [("force N", m["force_N"]), ("stroke mm", m["stroke_mm"]),
                          ("speed mm/s", m["speed_mms"]), ("hold N", m["holding_force_N"]),
                          ("SF", m["torque_margin"])])
    render_2x2(out_png, draw)
    return {"metrics": m, "warnings": r["warnings"]}


if __name__ == "__main__":
    print("linear:", analyze()["metrics"])
