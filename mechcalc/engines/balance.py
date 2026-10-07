"""balance.py — self-balancing two-wheel (inverted pendulum) robot. Righting-torque margin at a design
tilt, the max recoverable tilt for the given motors, the pendulum's natural timescale (how twitchy the
control must be), top speed, and mass. Metric keys match the balance_bot archetype.
"""
from __future__ import annotations

import math

from ._common import G, cfg, keybar, render_2x2

DEFAULTS = {
    "mass_kg": 1.0, "cog_height_mm": 120.0, "wheel_d_mm": 90.0, "motor_stall_torque_Nm": 0.5,
    "reduction": 1.0, "motor_rpm": 300.0, "design_tilt_deg": 15.0, "n_motors": 2,
}


def analyze(design: dict | None = None) -> dict:
    d = cfg(DEFAULTS, design)
    warn: list[str] = []
    h = d["cog_height_mm"] / 1000.0
    avail_wheel_torque = d["motor_stall_torque_Nm"] * d["reduction"] * d["n_motors"]
    # righting: wheel torque drives the base to catch a tilt. Required at design tilt = m*g*h*sin(theta)
    req_torque = d["mass_kg"] * G * h * math.sin(math.radians(d["design_tilt_deg"]))
    margin = avail_wheel_torque / req_torque if req_torque > 0 else float("inf")
    # max recoverable tilt from available torque
    ratio = avail_wheel_torque / (d["mass_kg"] * G * h) if h > 0 else 0.0
    tilt_recover = math.degrees(math.asin(min(1.0, ratio))) if ratio > 0 else 0.0
    nat_freq = 1.0 / (2 * math.pi) * math.sqrt(G / h) if h > 0 else 0.0    # inverted-pendulum timescale
    top_speed_kmh = d["motor_rpm"] * d["reduction"] / 60.0 * (math.pi * d["wheel_d_mm"] / 1000.0) * 3.6

    if margin < 1.5:
        warn.append(f"righting margin {margin:.2f} < 1.5 — it can't catch a {d['design_tilt_deg']:.0f}° tilt; "
                    f"stronger motors, more reduction, or a lower CoG (needs {req_torque:.3f} N·m)")
    if nat_freq > 3:
        warn.append(f"very twitchy ({nat_freq:.1f} Hz) — a low CoG falls fast; the control loop must run quick")
    if tilt_recover < 8:
        warn.append(f"only recovers ±{tilt_recover:.0f}° — knock it and it topples; more torque or lower CoG")
    return {
        "metrics": {
            "tilt_recover_deg": round(tilt_recover, 1),
            "torque_margin": round(margin, 2),
            "natural_freq_hz": round(nat_freq, 2),
            "top_speed_kmh": round(top_speed_kmh, 1),
            "mass_kg": round(d["mass_kg"], 2),
        },
        "warnings": warn,
        "_curves": {"d": d, "h": h, "avail": avail_wheel_torque, "req": req_torque},
    }


def plot(design: dict | None, out_png) -> dict:
    import numpy as np
    r = analyze(design)
    c = r["_curves"]
    m = r["metrics"]
    d = c["d"]

    def draw(ax):
        tilts = np.linspace(1, 45, 100)
        req = d["mass_kg"] * G * c["h"] * np.sin(np.radians(tilts))
        ax[0, 0].plot(tilts, req, lw=2); ax[0, 0].axhline(c["avail"], ls="--", c="g", label=f"motors {c['avail']:.2f} N·m")
        ax[0, 0].axvline(d["design_tilt_deg"], ls=":", c="#888")
        ax[0, 0].set_xlabel("tilt (deg)"); ax[0, 0].set_ylabel("righting torque (N·m)")
        ax[0, 0].set_title("Righting torque vs tilt"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
        hs = np.linspace(40, 300, 100) / 1000.0
        nf = 1.0 / (2 * math.pi) * np.sqrt(G / hs)
        ax[0, 1].plot(hs * 1000, nf, lw=2, color="#ec4899"); ax[0, 1].axvline(d["cog_height_mm"], ls=":", c="g")
        ax[0, 1].set_xlabel("CoG height (mm)"); ax[0, 1].set_ylabel("natural freq (Hz)")
        ax[0, 1].set_title("Twitchiness vs CoG height"); ax[0, 1].grid(alpha=.3)
        ax[1, 0].bar(["recover °", "torque SF"], [m["tilt_recover_deg"], min(m["torque_margin"], 4)], color=["#14b8a6", "#8b5cf6"])
        ax[1, 0].set_title("Margins"); ax[1, 0].grid(alpha=.3)
        keybar(ax[1, 1], [("recover °", m["tilt_recover_deg"]), ("torque SF", m["torque_margin"]),
                          ("freq Hz", m["natural_freq_hz"]), ("top km/h", m["top_speed_kmh"]),
                          ("mass kg", m["mass_kg"])])
    render_2x2(out_png, draw)
    return {"metrics": m, "warnings": r["warnings"]}


if __name__ == "__main__":
    print("balance:", analyze()["metrics"])
