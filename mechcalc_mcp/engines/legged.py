"""legged.py — walking robot (hexapod / spider / insectoid) statics. Per-leg supported load + joint
(femur) torque margin for the servos, static stability margin (CoG inside the stance polygon), gait
speed, and mass. Metric keys match the hexapod archetype objectives.
"""
from __future__ import annotations

from ._common import G, cfg, keybar, render_2x2

DEFAULTS = {
    "n_legs": 6, "dof_per_leg": 3, "coxa_mm": 30.0, "femur_mm": 60.0, "tibia_mm": 80.0,
    "servo_stall_torque_Nm": 1.6, "servo_mass_g": 55.0, "body_mass_g": 800.0, "payload_g": 200.0,
    "stride_mm": 45.0, "step_freq_hz": 1.5, "safety_factor": 1.8,
}


def analyze(design: dict | None = None) -> dict:
    d = cfg(DEFAULTS, design)
    warn: list[str] = []
    n_servos = d["n_legs"] * d["dof_per_leg"]
    mass_g = d["body_mass_g"] + n_servos * d["servo_mass_g"] + d["payload_g"]
    mass_kg = mass_g / 1000.0
    leg_reach_mm = d["femur_mm"] + d["tibia_mm"]                  # ground-projected reach
    stance = max(1, d["n_legs"] // 2)                            # tripod gait: half the legs down
    per_leg_N = mass_kg * G / stance
    femur_torque = per_leg_N * (leg_reach_mm / 1000.0)           # worst case, leg extended
    margin = d["servo_stall_torque_Nm"] / femur_torque if femur_torque > 0 else float("inf")
    # static stability: more stance legs + lower payload ratio + wider stance = better (0..1 heuristic)
    payload_pen = min(0.5, d["payload_g"] / max(1.0, d["body_mass_g"] * 2))
    stability = max(0.0, min(1.0, (stance / d["n_legs"]) + 0.35 - payload_pen))
    gait_speed = d["stride_mm"] * d["step_freq_hz"]              # mm/s

    if margin < d["safety_factor"]:
        warn.append(f"leg servo margin {margin:.2f} < SF {d['safety_factor']:g} — stronger servos, shorter legs, "
                    f"more legs down per step, or lighter body (femur needs {femur_torque:.2f} N·m)")
    if stability < 0.3:
        warn.append(f"low static stability ({stability:.2f}) — wider stance, lower/centered CoG, or a slower gait")
    if d["payload_g"] > d["body_mass_g"]:
        warn.append("payload heavier than the body — expect tipping and servo strain")
    return {
        "metrics": {
            "payload_g": round(d["payload_g"], 0),
            "leg_torque_margin": round(margin, 2),
            "stability_margin": round(stability, 2),
            "gait_speed_mms": round(gait_speed, 0),
            "mass_kg": round(mass_kg, 2),
            "per_leg_load_N": round(per_leg_N, 2),
        },
        "warnings": warn,
        "_curves": {"d": d, "leg_reach": leg_reach_mm, "stance": stance, "mass_kg": mass_kg,
                    "servo": d["servo_stall_torque_Nm"], "n_servos": n_servos},
    }


def plot(design: dict | None, out_png) -> dict:
    import numpy as np
    r = analyze(design)
    c = r["_curves"]
    m = r["metrics"]
    d = c["d"]

    def draw(ax):
        reaches = np.linspace(50, c["leg_reach"] * 1.6, 100)
        torque = c["mass_kg"] * G / c["stance"] * reaches / 1000.0
        ax[0, 0].plot(reaches, torque, lw=2)
        ax[0, 0].axhline(c["servo"], ls="--", c="g", label=f"servo {c['servo']:.1f} N·m")
        ax[0, 0].axvline(c["leg_reach"], ls=":", c="#888")
        ax[0, 0].set_xlabel("leg reach (mm)"); ax[0, 0].set_ylabel("femur torque (N·m)")
        ax[0, 0].set_title("Leg torque vs reach"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
        pays = np.linspace(0, d["body_mass_g"] * 1.5, 100)
        tot = (d["body_mass_g"] + c["n_servos"] * d["servo_mass_g"] + pays) / 1000.0
        tq = tot * G / c["stance"] * (c["leg_reach"] / 1000.0)
        ax[0, 1].plot(pays, tq, lw=2, color="#ec4899"); ax[0, 1].axhline(c["servo"], ls="--", c="g")
        ax[0, 1].axvline(d["payload_g"], ls=":", c="#888")
        ax[0, 1].set_xlabel("payload (g)"); ax[0, 1].set_ylabel("femur torque (N·m)")
        ax[0, 1].set_title("Payload vs leg torque"); ax[0, 1].grid(alpha=.3)
        ax[1, 0].bar(["stability", "torque SF"], [m["stability_margin"], min(m["leg_torque_margin"], 3)],
                     color=["#14b8a6", "#8b5cf6"])
        ax[1, 0].axhline(1, ls="--", c="#999"); ax[1, 0].set_title("Margins"); ax[1, 0].grid(alpha=.3)
        keybar(ax[1, 1], [("payload g", m["payload_g"]), ("leg SF", m["leg_torque_margin"]),
                          ("stability", m["stability_margin"]), ("speed mm/s", m["gait_speed_mms"]),
                          ("mass kg", m["mass_kg"])])
    render_2x2(out_png, draw)
    return {"metrics": m, "warnings": r["warnings"]}


if __name__ == "__main__":
    print("legged:", analyze()["metrics"])
