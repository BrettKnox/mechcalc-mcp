"""arm.py — robot-arm / manipulator statics. Reach, payload, tip deflection under load (beam theory,
tube links), and shoulder joint-torque margin (worst case: arm horizontal, fully extended). Covers the
3-DOF desktop arm and the pan-tilt turret. Metric keys match the robot_arm archetype objectives.
"""
from __future__ import annotations


from ._common import (G, E_of, bending_stress_mpa, cantilever_defl_mm, cfg, first_mode_hz, keybar,
                      render_2x2, rho_of, safety_factor, tube_I_mm4, tube_area_mm2)

DEFAULTS = {
    "dof": 3, "link1_mm": 200.0, "link2_mm": 180.0, "link_od_mm": 20.0, "link_wall_mm": 2.0,
    "link_material": "pla", "payload_g": 250.0, "motor_stall_torque_Nm": 1.8, "reduction": 5.0,
    "motor_mass_g": 120.0, "safety_factor": 2.0,
    # the environment's polymer-stiffness derate (1.0 = 20 C baseline); see engines/_common.apply_derate
    "env_E_scale": 1.0,
    # Everything past the shoulder except the links and the payload: [[mass_g, radius_mm], ...] from the
    # shoulder axis. None places the dof's own motors: the elbow motor at link1, any further (wrist)
    # motors at the tip. These used to count toward total mass and nothing else, so every servo hung
    # off the end of the arm loaded the shoulder for free.
    "distal_masses": None,
}


def _distal(d: dict, reach: float) -> list[tuple[float, float]]:
    """(kg, mm) point masses carried by the shoulder. Radii clamp to the reach they cannot exceed."""
    given = d.get("distal_masses")
    if given is None:
        dof = int(d["dof"])
        given = ([(d["motor_mass_g"], d["link1_mm"])] if dof >= 3 else []) + \
                [(d["motor_mass_g"], reach)] * max(dof - 3, 0)
    return [(float(g) / 1000.0, min(max(float(r), 0.0), reach)) for g, r in given]


def analyze(design: dict | None = None) -> dict:
    d = cfg(DEFAULTS, design)
    warn: list[str] = []
    reach = d["link1_mm"] + d["link2_mm"]                    # mm, fully extended
    I = tube_I_mm4(d["link_od_mm"], d["link_wall_mm"])
    # env_E_scale carries the environment's polymer-stiffness derate (1.0 = 20 C baseline). Without
    # it a 55 C sunlit arm reported exactly the same deflection as a bench one.
    E = E_of(d["link_material"]) * float(d.get("env_E_scale", 1.0) or 1.0)
    area_mm2 = tube_area_mm2(d["link_od_mm"], d["link_wall_mm"])
    link_len_m = reach / 1000.0
    link_mass = area_mm2 / 1e6 * link_len_m * rho_of(d["link_material"])   # kg, both links summed by total length
    distal = _distal(d, reach)
    distal_mass = sum(m for m, _ in distal)
    # motors at or below the shoulder (base yaw, the shoulder itself) carry nothing of the arm
    proximal_motors = min(int(d["dof"]), 2) * d["motor_mass_g"] / 1000.0
    motor_mass = (int(d["dof"]) * d["motor_mass_g"] / 1000.0 if d.get("distal_masses") is None
                  else proximal_motors + distal_mass)
    mass = link_mass + motor_mass
    payload_kg = d["payload_g"] / 1000.0
    # tip deflection: cantilever, point load = payload + half the link self-weight, at the tip, plus each
    # distal mass at its own radius: P*a^2*(3L - a)/(6EI)
    P_tip = (payload_kg + 0.5 * link_mass) * G
    defl = cantilever_defl_mm(P_tip, reach, E, I) + sum(
        m * G * a ** 2 * (3 * reach - a) / (6 * E * I) for m, a in distal) if I > 0 else float("inf")
    # shoulder torque required (arm horizontal): payload at reach + link CoG at reach/2 + distal masses
    distal_moment = sum(m * G * a / 1000.0 for m, a in distal)                        # N*m
    req_torque = payload_kg * G * (reach / 1000.0) + link_mass * G * (reach / 2000.0) + distal_moment
    avail_torque = d["motor_stall_torque_Nm"] * d["reduction"]
    margin = avail_torque / req_torque if req_torque > 0 else float("inf")

    if margin < d["safety_factor"]:
        warn.append(f"shoulder torque margin {margin:.2f} < SF {d['safety_factor']:g} — bigger motor, more "
                    f"reduction, shorter links, or lighter payload (needs {req_torque:.2f} N·m, have {avail_torque:.2f})")
    if defl > reach * 0.01:
        warn.append(f"tip sags {defl:.1f} mm ({defl / reach * 100:.1f}% of reach) — stiffer link (bigger OD/"
                    f"wall or metal/carbon) for precision")
    # root bending stress and margin: deflection alone never showed whether the link would YIELD. The root
    # moment of a horizontal arm IS the shoulder torque, so it carries the same distal masses.
    root_stress = bending_stress_mpa(req_torque * 1000.0, d["link_od_mm"] / 2.0, I)
    sf = safety_factor(root_stress, d["link_material"], across_layers=True)
    if sf < d["safety_factor"]:
        warn.append(f"link root stress {root_stress:.1f} MPa gives SF {sf:.2f} < {d['safety_factor']:g} "
                    f"(printed across layers) — thicker wall, bigger OD, or a stronger material")
    return {
        "metrics": {
            "reach_mm": round(reach, 1),
            "payload_g": round(d["payload_g"], 0),
            "tip_deflection_mm": round(defl, 2),
            "joint_torque_margin": round(margin, 2),
            "mass_kg": round(mass, 2),
            "shoulder_torque_Nm": round(req_torque, 2),
            "avail_torque_Nm": round(avail_torque, 2),
            "link_stiffness_Nm2": round(E * I / 1e6, 1),
            "root_stress_mpa": root_stress,
            "link_safety_factor": sf,
            "first_mode_hz": first_mode_hz(E, I, mass, reach),
        },
        "warnings": warn,
        "_curves": {"reach": reach, "E": E, "I": I, "P_tip": P_tip, "payload_kg": payload_kg,
                    "link_mass": link_mass, "avail": avail_torque, "motor_mass": motor_mass,
                    "distal_moment": distal_moment},
    }


def plot(design: dict | None, out_png) -> dict:
    import numpy as np
    r = analyze(design)
    c = r["_curves"]
    m = r["metrics"]

    def draw(ax):
        Ls = np.linspace(80, c["reach"] * 1.4, 100)
        defl = c["P_tip"] * Ls ** 3 / (3 * c["E"] * c["I"])
        ax[0, 0].plot(Ls, defl, lw=2)
        ax[0, 0].axvline(c["reach"], ls=":", c="g", label=f"reach {c['reach']:.0f} mm")
        ax[0, 0].axhline(c["reach"] * 0.01, ls="--", c="#d33", label="1% tolerance")
        ax[0, 0].set_xlabel("reach (mm)"); ax[0, 0].set_ylabel("tip deflection (mm)")
        ax[0, 0].set_title("Stiffness — tip sag vs reach"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
        pays = np.linspace(0, max(c["payload_kg"] * 2, 0.5), 100)
        req = pays * G * (c["reach"] / 1000.0) + c["link_mass"] * G * (c["reach"] / 2000.0) + c["distal_moment"]
        ax[0, 1].plot(pays * 1000, req, lw=2, label="required")
        ax[0, 1].axhline(c["avail"], ls="--", c="g", label=f"motor {c['avail']:.1f} N·m")
        ax[0, 1].axvline(c["payload_kg"] * 1000, ls=":", c="#888")
        ax[0, 1].set_xlabel("payload (g)"); ax[0, 1].set_ylabel("shoulder torque (N·m)")
        ax[0, 1].set_title("Payload vs torque"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3)
        ax[1, 0].bar(["links", "motors"], [c["link_mass"], c["motor_mass"]], color=["#8b5cf6", "#0ea5e9"])
        ax[1, 0].set_ylabel("mass (kg)"); ax[1, 0].set_title("Mass breakdown"); ax[1, 0].grid(alpha=.3)
        keybar(ax[1, 1], [("reach cm", m["reach_mm"] / 10), ("payload g", m["payload_g"]),
                          ("sag mm", m["tip_deflection_mm"]), ("torque SF", m["joint_torque_margin"]),
                          ("mass kg", m["mass_kg"])])
    render_2x2(out_png, draw)
    return {"metrics": m, "warnings": r["warnings"]}


if __name__ == "__main__":
    a = analyze()
    print("arm:", a["metrics"])
    for w in a["warnings"]:
        print("  warn:", w)
