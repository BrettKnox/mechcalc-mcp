"""vacuum.py — the pneumatics behind a suction or granular-jamming gripper.

How long to pump down, how long to let go, and how hard it can hold. Nothing else in the app has a
fluid domain, so this is the whole of it.

Three things this engine is deliberately careful about:

**It models suction only.** A granular jamming gripper holds by three separate mechanisms (Brown,
Rodenberg, Amend, Jaeger & Lipson, PNAS 2010): geometric interlocking, static friction, and vacuum
suction wherever the membrane seals against the object. Only the third is pressure times area. The
other two depend on the object's shape and on the membrane's hyperelastic behaviour, which
`matprops` cannot describe and is not going to be made to guess. So `suction_force_N` is the part of
the grip this engine can defend, and the total is a bench measurement, not a computed number.

**A pump cannot beat its own ultimate.** Pumping speed falls to zero as the chamber approaches the
pump's ultimate pressure, so the honest model is dP/dt = -(S/V)(P - P_ult), which sends the time to
infinity rather than quietly reporting a target the hardware can never reach.

**The filter is not optional.** Grains reaching a diaphragm pump destroy it, and it is the failure
every published jamming gripper hits first. A design with no filter is warned about by name.
"""
from __future__ import annotations

import math

from ._common import cfg, keybar, render_2x2

ATM_KPA = 101.3
AIR_VISCOSITY_PAS = 1.81e-5           # dry air at 20 C

DEFAULTS = {
    "pump_flow_lpm": 12.0,            # free-air displacement off the pump's own label
    "pump_ultimate_kPa": 85.0,        # the best GAUGE vacuum it can pull, also off the label
    "target_vacuum_kPa": 70.0,        # what the gripper needs to jam
    "chamber_ml": 75.0,               # bowl + membrane slack + the dead volume in the line
    "tube_id_mm": 4.0,
    "tube_len_mm": 900.0,
    "membrane_dia_mm": 60.0,
    # What fraction of the membrane's face actually seals against a real object. 1.0 would be a flat
    # plate on flat glass. 0.10 is the low end of the published range for a head this size and is
    # ASSUMED until `calibrate` has a measured grip -- it is the least defensible number here.
    "seal_fraction": 0.10,
    "valve_release_ms": 30.0,         # how fast the solenoid dumps the chamber back to atmosphere
    "has_filter": 1.0,
    "duty_cycle_pct": 30.0,           # diaphragm pumps this size are not rated continuous
}


def _tube_conductance_lps(id_mm: float, len_mm: float, mean_abs_kpa: float) -> float:
    """Poiseuille conductance of a round tube in viscous flow, L/s.

    Rough vacuum at these diameters is viscous, not molecular, so this is the right regime. It
    almost always comes out enormous next to a hobby pump -- which is the useful answer: the tubing
    is not the bottleneck, and no amount of shorter hose buys anything.
    """
    if id_mm <= 0 or len_mm <= 0:
        return float("inf")
    d, ln = id_mm / 1000.0, len_mm / 1000.0
    c_m3s = (math.pi * d ** 4 * (mean_abs_kpa * 1000.0)) / (128.0 * AIR_VISCOSITY_PAS * ln)
    return c_m3s * 1000.0


def analyze(design: dict | None = None) -> dict:
    d = cfg(DEFAULTS, design)
    warn: list[str] = []

    target_gauge = max(0.0, float(d["target_vacuum_kPa"]))
    ultimate_gauge = max(0.0, float(d["pump_ultimate_kPa"]))
    p_target_abs = ATM_KPA - target_gauge
    p_ult_abs = ATM_KPA - ultimate_gauge
    volume_l = max(1e-6, float(d["chamber_ml"]) / 1000.0)

    mean_abs = (ATM_KPA + p_target_abs) / 2.0
    conductance = _tube_conductance_lps(d["tube_id_mm"], d["tube_len_mm"], mean_abs)
    pump_lps = max(1e-9, float(d["pump_flow_lpm"]) / 60.0)
    eff_lps = pump_lps if conductance == float("inf") else 1.0 / (1.0 / pump_lps + 1.0 / conductance)
    line_loss_pct = (1.0 - eff_lps / pump_lps) * 100.0

    # dP/dt = -(S/V)(P - P_ult)  ->  t = (V/S) ln((P0 - P_ult)/(P_target - P_ult))
    reachable = target_gauge < ultimate_gauge
    if reachable:
        pumpdown = (volume_l / eff_lps) * math.log((ATM_KPA - p_ult_abs) / (p_target_abs - p_ult_abs))
    else:
        pumpdown = float("inf")
        warn.append(
            f"the pump cannot reach the vacuum this gripper needs: it pulls {ultimate_gauge:g} kPa "
            f"and {target_gauge:g} kPa is asked for. No pumping time fixes this — a stronger pump, "
            f"or a design that jams at a lower vacuum.")

    # Letting go is the other half of a pick-and-place, and it is valve-limited rather than
    # tube-limited: air rushes back in through the same wide pipe it was pulled out of.
    tube_refill_s = volume_l / conductance if conductance not in (0, float("inf")) else 0.0
    release_s = max(float(d["valve_release_ms"]) / 1000.0, tube_refill_s)

    area_m2 = math.pi * (float(d["membrane_dia_mm"]) / 2000.0) ** 2
    seal = min(1.0, max(0.0, float(d["seal_fraction"])))
    suction_max_n = target_gauge * 1000.0 * area_m2
    suction_n = suction_max_n * seal

    cycle_s = (pumpdown + release_s) if reachable else float("inf")

    if not float(d["has_filter"]):
        warn.append("NO INLINE FILTER: grains reaching the pump destroy it, and this is the failure "
                    "every published jamming gripper hits first. It is not an optional component.")
    if reachable and pumpdown > 3.0:
        warn.append(f"pump-down takes {pumpdown:.1f} s — slow for a pick-and-place. A bigger pump or "
                    f"less dead volume in the line; the chamber is {d['chamber_ml']:g} mL.")
    if line_loss_pct > 25.0:
        warn.append(f"the line is throttling the pump by {line_loss_pct:.0f}% — {d['tube_len_mm']:g} mm "
                    f"of {d['tube_id_mm']:g} mm bore. Wider or shorter.")
    if reachable and target_gauge > ultimate_gauge * 0.9:
        warn.append(f"asking for {target_gauge:g} kPa from a pump that stops at {ultimate_gauge:g} kPa "
                    f"is the slow end of its curve; a small leak will lose the grip entirely.")
    if float(d["duty_cycle_pct"]) < 100 and reachable and cycle_s > 0:
        warn.append(f"the pump is rated {d['duty_cycle_pct']:g}% duty, so it must be valved off between "
                    f"picks rather than left running — check the chamber holds vacuum with it stopped.")
    # Said every time, not just when something looks wrong: a number this partial reads as a total.
    warn.append(f"suction accounts for {suction_n:.1f} N of the grip. Interlocking and friction are "
                f"NOT modelled — a latex membrane is hyperelastic and nothing here can describe it. "
                f"Total grip force is a bench measurement, and seal_fraction={seal:g} is assumed.")

    return {
        "metrics": {
            "pumpdown_s": round(pumpdown, 2) if reachable else None,
            "release_s": round(release_s, 3),
            "cycle_s": round(cycle_s, 2) if reachable else None,
            "suction_force_N": round(suction_n, 1),
            "suction_max_N": round(suction_max_n, 1),
            "vacuum_margin_kPa": round(ultimate_gauge - target_gauge, 1),
            "effective_speed_lpm": round(eff_lps * 60.0, 2),
            "line_loss_pct": round(line_loss_pct, 2),
            "sealed_area_cm2": round(area_m2 * 1e4 * seal, 2),
        },
        "warnings": warn,
        "_curves": {"d": d, "p_ult_abs": p_ult_abs, "eff_lps": eff_lps, "volume_l": volume_l,
                    "reachable": reachable, "conductance": conductance},
    }


def plot(design: dict | None, out_png) -> dict:
    import numpy as np
    r = analyze(design)
    c, m, d = r["_curves"], r["metrics"], r["_curves"]["d"]

    def draw(ax):
        # 1. the pump-down curve itself, and why it asymptotes
        if c["reachable"]:
            t_end = max(0.5, float(m["pumpdown_s"]) * 1.8)
            t = np.linspace(0, t_end, 200)
            p_abs = c["p_ult_abs"] + (ATM_KPA - c["p_ult_abs"]) * np.exp(-c["eff_lps"] / c["volume_l"] * t)
            ax[0, 0].plot(t, ATM_KPA - p_abs, lw=2, color="#0ea5e9")
            ax[0, 0].axhline(d["target_vacuum_kPa"], ls="--", c="#16a34a",
                             label=f"target {d['target_vacuum_kPa']:g} kPa")
            ax[0, 0].axvline(m["pumpdown_s"], ls=":", c="#16a34a")
        ax[0, 0].axhline(d["pump_ultimate_kPa"], ls="--", c="#d33",
                         label=f"pump limit {d['pump_ultimate_kPa']:g} kPa")
        ax[0, 0].set_xlabel("time (s)"); ax[0, 0].set_ylabel("gauge vacuum (kPa)")
        ax[0, 0].set_title("Pump-down"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)

        # 2. how much longer a bigger chamber costs
        vols = np.linspace(20, 300, 100)
        if c["reachable"]:
            ratio = math.log((ATM_KPA - c["p_ult_abs"]) /
                             ((ATM_KPA - d["target_vacuum_kPa"]) - c["p_ult_abs"]))
            ax[0, 1].plot(vols, (vols / 1000.0) / c["eff_lps"] * ratio, lw=2, color="#8b5cf6")
            ax[0, 1].axvline(d["chamber_ml"], ls=":", c="g")
        ax[0, 1].set_xlabel("chamber + line volume (mL)"); ax[0, 1].set_ylabel("pump-down (s)")
        ax[0, 1].set_title("Dead volume is the cost"); ax[0, 1].grid(alpha=.3)

        # 3. what is modelled against what is not
        ax[1, 0].bar(["suction\n(modelled)", "if fully\nsealed"],
                     [m["suction_force_N"], m["suction_max_N"]], color=["#8b5cf6", "#cbd5e1"])
        ax[1, 0].set_ylabel("N")
        ax[1, 0].set_title("Interlocking + friction NOT included"); ax[1, 0].grid(alpha=.3)

        keybar(ax[1, 1], [("pumpdown s", m["pumpdown_s"] if m["pumpdown_s"] is not None else 0),
                          ("release s", m["release_s"]),
                          ("suction N", m["suction_force_N"]),
                          ("margin kPa", m["vacuum_margin_kPa"]),
                          ("line loss %", m["line_loss_pct"])])
    render_2x2(out_png, draw)
    return {"metrics": m, "warnings": r["warnings"]}


if __name__ == "__main__":
    r = analyze()
    print("vacuum:", r["metrics"])
    m = r["metrics"]
    assert 0.3 < m["pumpdown_s"] < 2.0, m           # 75 mL on a 12 L/min pump is under a second
    assert m["line_loss_pct"] < 2.0, "4 mm bore is nowhere near the bottleneck"
    assert m["vacuum_margin_kPa"] == 15.0, m
    assert any("NOT modelled" in w for w in r["warnings"]), "the partial answer is not flagged"

    # A pump that cannot reach the target must say so rather than quote a time
    bad = analyze({"pump_ultimate_kPa": 50.0, "target_vacuum_kPa": 70.0})
    assert bad["metrics"]["pumpdown_s"] is None
    assert any("cannot reach" in w for w in bad["warnings"]), bad["warnings"]

    # No filter is a named failure, not a silent one
    assert any("NO INLINE FILTER" in w for w in analyze({"has_filter": 0})["warnings"])

    # A long thin line DOES throttle, or the conductance model is not doing anything
    thin = analyze({"tube_id_mm": 1.0, "tube_len_mm": 2000.0})
    assert thin["metrics"]["line_loss_pct"] > 25.0, thin["metrics"]
    print("vacuum: self-check ok")
