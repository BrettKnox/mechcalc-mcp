"""cnc.py — CNC / gantry machine rigidity + accuracy. Cutting force from the material, gantry deflection
under that force (simply-supported beam), positioning error (deflection + drive backlash + step
resolution), max feed, and work volume. Covers the CoreXY plotter -> CNC. Metric keys match the
cnc_machine archetype objectives.
"""
from __future__ import annotations

from ._common import E_of, cfg, keybar, render_2x2, simply_supported_defl_mm, tube_I_mm4

# specific cutting force kc (N/mm^2) — how hard the material fights the tool
KC = {"foam": 5, "wood": 40, "mdf": 30, "acrylic": 60, "pcb": 80, "aluminum": 800,
      "brass": 700, "steel": 2000, "pla": 45}
BACKLASH_UM = {"leadscrew": 10.0, "ballscrew": 3.0, "belt": 60.0, "rack": 40.0}

# A stepper's quoted torque is its HOLDING torque, measured at standstill. Back-EMF eats it as speed
# rises, and the corner is low — a 42 mm NEMA 17 on 24 V is already falling off by ~300 rpm. Sizing a
# cut against the holding figure while quoting a feed rate at 600 rpm is the classic way to buy a
# motor that stalls the first time it moves: it is the same motor being asked for torque it only has
# when stopped.
#
# ponytail: flat-then-1/rpm, with the supply voltage as the knob because that is the lever that
# actually moves the corner. The truth is the motor's own torque-speed curve — pass the real figure
# as `stepper_torque_at_feed_Nm` and this is skipped.
STEPPER_CORNER_RPM_AT_24V = 300.0


def torque_at_speed(holding_nm: float, rpm: float, supply_v: float = 24.0) -> float:
    """Usable torque at `rpm`. Flat to the corner speed, then falls as corner/rpm."""
    if rpm <= 0 or holding_nm <= 0:
        return max(0.0, holding_nm)
    corner = STEPPER_CORNER_RPM_AT_24V * max(supply_v, 1.0) / 24.0
    return holding_nm * min(1.0, corner / rpm)

DEFAULTS = {
    "span_x_mm": 300.0, "span_y_mm": 300.0, "span_z_mm": 100.0,
    "gantry_od_mm": 40.0, "gantry_wall_mm": 3.0, "gantry_material": "aluminum",
    "spindle_power_W": 500.0, "drive": "leadscrew", "leadscrew_pitch_mm": 8.0,
    "stepper_torque_Nm": 0.45, "stepper_max_rpm": 600.0, "cut_material": "aluminum",
    "doc_mm": 0.5, "chipload_mm": 0.05, "accuracy_target_um": 100.0, "drivetrain_eff": 0.85,
    "env_E_scale": 1.0, "supply_v": 24.0, "stepper_torque_at_feed_Nm": None,
}


def analyze(design: dict | None = None) -> dict:
    import math
    d = cfg(DEFAULTS, design)
    warn: list[str] = []
    work_volume_l = d["span_x_mm"] * d["span_y_mm"] * d["span_z_mm"] / 1e6
    kc = KC.get((d["cut_material"] or "").lower(), 800)
    cut_force = kc * d["doc_mm"] * d["chipload_mm"]                # N (chip cross-section x kc)
    I = tube_I_mm4(d["gantry_od_mm"], d["gantry_wall_mm"])
    E = E_of(d["gantry_material"]) * float(d.get("env_E_scale", 1.0) or 1.0)
    gantry_defl_um = simply_supported_defl_mm(cut_force, d["span_x_mm"], E, I) * 1000.0
    step_res_um = d["leadscrew_pitch_mm"] / (200 * 16) * 1000.0    # 200 steps x 16 microstep
    backlash = BACKLASH_UM.get((d["drive"] or "").lower(), 10.0)
    pos_error_um = gantry_defl_um + backlash + step_res_um
    if (d["drive"] or "").lower() == "leadscrew":
        max_feed = d["stepper_max_rpm"] / 60.0 * d["leadscrew_pitch_mm"]
    else:
        max_feed = d["stepper_max_rpm"] / 60.0 * 40.0             # ~40 mm/rev for a GT2 pulley
    # can the stepper drive the cut force through the screw?
    screw_torque_needed = cut_force * (d["leadscrew_pitch_mm"] / 1000.0) / (2 * math.pi * d["drivetrain_eff"])
    if gantry_defl_um > d["accuracy_target_um"]:
        warn.append(f"gantry flexes {gantry_defl_um:.0f} µm under {cut_force:.0f} N cut > {d['accuracy_target_um']:.0f} µm "
                    f"target — stiffer/bigger gantry, shorter span, or lighter cuts")
    # The cut happens at the feed, not at standstill, so the check has to use the torque available
    # there. Checking against holding torque passes cuts that stall the machine.
    measured = d.get("stepper_torque_at_feed_Nm")
    if measured:
        torque_at_feed, torque_src = float(measured), "measured"
    else:
        torque_at_feed = torque_at_speed(d["stepper_torque_Nm"], d["stepper_max_rpm"], d["supply_v"])
        torque_src = "estimated"
    if screw_torque_needed > torque_at_feed:
        warn.append(f"stepper can't push the cut at feed: needs {screw_torque_needed:.2f} N·m, has "
                    f"{torque_at_feed:.2f} at {d['stepper_max_rpm']:.0f} rpm on {d['supply_v']:.0f} V "
                    f"({d['stepper_torque_Nm']:.2f} holding) — stiffer supply voltage, coarser screw, "
                    f"lighter cut, or a bigger motor")
    if torque_src == "estimated" and torque_at_feed < d["stepper_torque_Nm"]:
        warn.append(f"the {torque_at_feed:.2f} N·m figure is an estimated derate of the "
                    f"{d['stepper_torque_Nm']:.2f} N·m holding torque, not a measurement — read it off "
                    f"the motor's torque-speed curve and pass stepper_torque_at_feed_Nm to check it "
                    f"properly")
    if (d["drive"] or "").lower() == "belt":
        warn.append("belt drive: fast but springy — fine for a plotter/laser, marginal for metal")
    return {
        "metrics": {
            "gantry_deflection_um": round(gantry_defl_um, 1),
            "work_volume_l": round(work_volume_l, 2),
            "positioning_error_um": round(pos_error_um, 1),
            "max_feed_mm_s": round(max_feed, 1),
            "spindle_power_W": round(d["spindle_power_W"], 0),
            "cutting_force_N": round(cut_force, 1),
            "stepper_torque_at_feed_Nm": round(torque_at_feed, 3),
            "screw_torque_needed_Nm": round(screw_torque_needed, 3),
        },
        "warnings": warn,
        "_curves": {"E": E, "I": I, "cut_force": cut_force, "span": d["span_x_mm"],
                    "target": d["accuracy_target_um"], "backlash": backlash, "step_res": step_res_um},
    }


def plot(design: dict | None, out_png) -> dict:
    import numpy as np
    r = analyze(design)
    c = r["_curves"]
    m = r["metrics"]

    def draw(ax):
        spans = np.linspace(100, c["span"] * 1.5, 100)
        defl = c["cut_force"] * spans ** 3 / (48 * c["E"] * c["I"]) * 1000.0
        ax[0, 0].plot(spans, defl, lw=2)
        ax[0, 0].axvline(c["span"], ls=":", c="g", label=f"span {c['span']:.0f} mm")
        ax[0, 0].axhline(c["target"], ls="--", c="#d33", label=f"target {c['target']:.0f} µm")
        ax[0, 0].set_xlabel("gantry span (mm)"); ax[0, 0].set_ylabel("deflection (µm)")
        ax[0, 0].set_title("Rigidity — gantry sag vs span"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
        forces = np.linspace(0, max(c["cut_force"] * 3, 30), 100)
        defl2 = forces * c["span"] ** 3 / (48 * c["E"] * c["I"]) * 1000.0
        ax[0, 1].plot(forces, defl2, lw=2, color="#d33")
        ax[0, 1].axvline(c["cut_force"], ls=":", c="#888", label=f"cut {c['cut_force']:.0f} N")
        ax[0, 1].set_xlabel("cutting force (N)"); ax[0, 1].set_ylabel("deflection (µm)")
        ax[0, 1].set_title("Deflection vs cut load"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3)
        ax[1, 0].bar(["gantry", "backlash", "steps"], [m["gantry_deflection_um"], c["backlash"], c["step_res"]], color="#8b5cf6")
        ax[1, 0].set_ylabel("µm"); ax[1, 0].set_title("Positioning error budget"); ax[1, 0].grid(alpha=.3)
        keybar(ax[1, 1], [("gantry µm", m["gantry_deflection_um"]), ("pos µm", m["positioning_error_um"]),
                          ("vol L", m["work_volume_l"]), ("feed mm/s", m["max_feed_mm_s"]),
                          ("spindle W", m["spindle_power_W"])])
    render_2x2(out_png, draw)
    return {"metrics": m, "warnings": r["warnings"]}


if __name__ == "__main__":
    print("cnc:", analyze()["metrics"])
