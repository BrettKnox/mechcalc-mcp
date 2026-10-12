"""mechanism.py — the geometry between a straight-line ram and a rotating joint, and where it is worst.

Three things the design asserted rather than derived.

**The moment arm is not constant.** A ram pushing on a lever has a moment arm that goes to ZERO at
both ends of its travel, because at those points the ram line passes through the pivot. The design
sized the boom ram at one pose and assumed that pose was the worst one. It is not, and the actual
worst case can sit anywhere in the range depending on where the anchor and the boss are placed.

**Reach was declared, not computed.** `reach_mm = 320` is in the build type as a number somebody
typed. What two links of 180 and 140 mm can actually touch, given their joint limits, is a different
question with a different answer.

**Tip-over did not know the arm existed.** `analysis.py:162` computes `(track/2)/cog_height`, a
static number for a vehicle standing still. On a machine that digs, the centre of gravity swings out
on the end of an arm, and how close it gets to the edge of the wheelbase depends on which way the
arm is pointing. That model reduces to the old one when the arm is stowed, which is the test that it
is a generalisation rather than a different guess.
"""
from __future__ import annotations

import math

G = 9.81


# ── ram on a lever ───────────────────────────────────────────────────────────────────────────────

def ram(anchor_mm: float, boss_mm: float, angle_from_deg: float, angle_to_deg: float,
        *, samples: int = 181, force_N: float = 0.0, load_torque_Nm: float = 0.0) -> dict:
    """A linear actuator driving a joint through a lever, swept across the whole joint range.

    `anchor_mm` is from the joint axis to the ram's fixed end, `boss_mm` from the joint axis to the
    clevis boss on the moving link, and the angle is between those two arms.

    The moment arm is h = a*r*sin(theta) / L(theta), which is twice the triangle's area over its
    base. It vanishes at theta = 0 and theta = 180 -- the toggle points -- so a range that includes
    either has a pose where the ram cannot move the joint at all however hard it pushes.
    """
    a, r = float(anchor_mm), float(boss_mm)
    if a <= 0 or r <= 0:
        raise ValueError("anchor and boss distances must both be positive")
    lo, hi = sorted((float(angle_from_deg), float(angle_to_deg)))
    if hi - lo <= 0:
        raise ValueError("the joint range must span more than zero degrees")

    rows = []
    for i in range(max(3, samples)):
        th = math.radians(lo + (hi - lo) * i / (max(3, samples) - 1))
        L = math.sqrt(a * a + r * r - 2 * a * r * math.cos(th))         # law of cosines
        h = (a * r * math.sin(th) / L) if L > 1e-9 else 0.0             # 2*area / base
        rows.append({"angle_deg": round(math.degrees(th), 2),
                     "ram_length_mm": round(L, 3), "moment_arm_mm": round(h, 3)})

    lengths = [x["ram_length_mm"] for x in rows]
    arms = [x["moment_arm_mm"] for x in rows]
    stroke = max(lengths) - min(lengths)
    worst = min(rows, key=lambda x: x["moment_arm_mm"])
    best = max(rows, key=lambda x: x["moment_arm_mm"])

    out = {
        "anchor_mm": a, "boss_mm": r, "range_deg": [lo, hi],
        "stroke_mm": round(stroke, 2),
        "ram_length_min_mm": round(min(lengths), 2), "ram_length_max_mm": round(max(lengths), 2),
        "moment_arm_min_mm": round(min(arms), 3), "moment_arm_max_mm": round(max(arms), 3),
        "worst_angle_deg": worst["angle_deg"], "best_angle_deg": best["angle_deg"],
        "curve": rows, "warnings": [], "notes": [],
    }

    # A toggle inside the range is a dead point: infinite force, zero torque.
    if min(arms) < 0.05 * max(arms):
        out["warnings"].append(
            f"TOGGLE POINT inside the travel: the moment arm falls to {min(arms):.2f} mm at "
            f"{worst['angle_deg']:g} deg, against {max(arms):.1f} mm at its best. The ram cannot "
            f"move the joint there at any force. Move the anchor or the boss so the range avoids "
            f"0 and 180 degrees between them.")

    if force_N:
        torques = [force_N * x["moment_arm_mm"] / 1000.0 for x in rows]
        out["torque_min_Nm"] = round(min(torques), 3)
        out["torque_max_Nm"] = round(max(torques), 3)
        out["torque_at_worst_Nm"] = round(force_N * worst["moment_arm_mm"] / 1000.0, 3)
        if load_torque_Nm:
            margins = [t / load_torque_Nm if load_torque_Nm else float("inf") for t in torques]
            worst_i = min(range(len(margins)), key=lambda i: margins[i])
            out["margin_min"] = round(min(margins), 2)
            out["margin_worst_angle_deg"] = rows[worst_i]["angle_deg"]
            out["notes"].append(
                f"worst margin {min(margins):.2f}x at {rows[worst_i]['angle_deg']:g} deg, NOT at "
                f"either end of the travel -- sizing at full extension alone would have missed it"
                if 0 < worst_i < len(rows) - 1 else
                f"worst margin {min(margins):.2f}x at {rows[worst_i]['angle_deg']:g} deg, at the end "
                f"of the travel")
            if min(margins) < 1.0:
                out["warnings"].append(
                    f"THE RAM STALLS INSIDE ITS OWN RANGE: only {min(margins):.2f}x the torque needed "
                    f"at {rows[worst_i]['angle_deg']:g} deg. It will drive to there and stop.")
    return out


# ── workspace ────────────────────────────────────────────────────────────────────────────────────

def workspace(link1_mm: float, link2_mm: float, *, joint1_deg: float = 95.0,
              joint2_deg: float = 120.0, tool_mm: float = 0.0) -> dict:
    """What the arm can actually touch, from link lengths and joint limits.

    Reach was a declared number. Two links can only straighten as far as their elbow lets them, so
    the honest maximum is geometric and not a wish.
    """
    l1, l2 = float(link1_mm), float(link2_mm)
    t = float(tool_mm or 0.0)
    if l1 <= 0 or l2 <= 0:
        raise ValueError("both link lengths must be positive")

    straight = l1 + l2 + t
    # The elbow's limit stops it straightening if its range does not reach 180 degrees of included
    # angle. Modelled as: the elbow can open to at most `joint2_deg` from folded.
    open_deg = min(180.0, float(joint2_deg))
    reach = math.sqrt(l1 * l1 + (l2 + t) ** 2 - 2 * l1 * (l2 + t) * math.cos(math.radians(open_deg)))
    reach = min(reach, straight)
    folded = math.sqrt(l1 * l1 + (l2 + t) ** 2 - 2 * l1 * (l2 + t) * math.cos(math.radians(0.001)))

    out = {
        "link1_mm": l1, "link2_mm": l2, "tool_mm": t,
        "reach_if_straight_mm": round(straight, 1),
        "reach_mm": round(reach, 1),
        "closest_mm": round(max(abs(l1 - (l2 + t)), folded), 1),
        "shoulder_range_deg": float(joint1_deg), "elbow_range_deg": float(joint2_deg),
        "annulus_mm": [round(max(abs(l1 - (l2 + t)), folded), 1), round(reach, 1)],
        "warnings": [], "notes": [],
    }
    if reach < straight - 1.0:
        out["notes"].append(
            f"the elbow only opens {open_deg:g} deg, so the arm reaches {reach:.0f} mm rather than "
            f"the {straight:.0f} mm its links would allow straightened")
    return out


# ── tip-over, as a function of where the arm is pointing ─────────────────────────────────────────

def tipover(*, track_mm: float, wheelbase_mm: float, cog_height_mm: float,
            base_kg: float, arm_kg: float, payload_kg: float = 0.0,
            reach_mm: float = 0.0, slew_offset_mm: float = 0.0,
            counterweight_kg: float = 0.0, counterweight_r_mm: float = 100.0,
            counterweight_on: str = "house", counterweight_at_mm: float = 0.0,
            arm_cog_fraction: float = 0.5, samples: int = 73) -> dict:
    """Stability margin around the full slew circle, instead of one number for a parked vehicle.

    The support polygon is the rectangle through the four contact patches. Its boundary sits at
    d(psi) = min(halftrack/|sin psi|, halfbase/|cos psi|) from the centre. The centre of gravity
    moves out along the arm as it extends and swings with it, so what matters is how much of d(psi)
    is left.

    With the arm stowed this collapses to (track/2) / cog_height, which is exactly the old static
    formula -- the test that this generalises the previous answer rather than replacing it with a
    different guess.
    """
    if cog_height_mm <= 0:
        raise ValueError("centre-of-gravity height must be positive")
    hw, hl = float(track_mm) / 2.0, float(wheelbase_mm) / 2.0
    cw = float(counterweight_kg or 0.0)
    total = float(base_kg) + float(arm_kg) + float(payload_kg) + cw
    if total <= 0:
        raise ValueError("total mass must be positive")

    # How far the combined CoG sits from the slew axis when the arm points somewhere.
    #
    # TWO KINDS OF MASS, and only one of them turns. On an excavator the counterweight belongs to the
    # HOUSE: it swings with the arm and always sits diametrically opposite it, which is the whole
    # reason the lump is on the back of the cab rather than on the tracks. On a mobile manipulator
    # laid out the way every real one is -- a flat box base, the arm bolted to a bare top deck,
    # nothing beside the shoulder -- the ballast is a plate in the BASE and does not turn. It lowers
    # the centre of gravity instead of opposing the arm.
    #
    # Modelling the second as the first is not a small error: a fixed weight enters `offset`, gets
    # multiplied by cos(psi), and is credited with opposing the arm at every heading. Modelling the
    # first as the second is the error this function's original note warns about, and both are wrong
    # in the same way -- a moment that rotates and one that does not are not the same quantity.
    fixed = str(counterweight_on or "house").lower() == "base"
    r_arm = float(reach_mm) * float(arm_cog_fraction)
    m_rot = (float(arm_kg) * r_arm + float(payload_kg) * float(reach_mm)
             - (0.0 if fixed else cw * float(counterweight_r_mm)))
    m_fixed = (cw * float(counterweight_at_mm)) if fixed else 0.0
    offset = m_rot / total

    rows = []
    for i in range(max(4, samples)):
        psi = 2 * math.pi * i / max(4, samples)
        # CoG on the ground plane, measured from the middle of the support rectangle.
        cx = float(slew_offset_mm) + offset * math.cos(psi) + m_fixed / total
        cy = offset * math.sin(psi)
        # Distance from the CoG to the rectangle boundary along its own outward direction.
        s, c = math.sin(psi), math.cos(psi)
        d = min(hw / abs(s) if abs(s) > 1e-9 else float("inf"),
                hl / abs(c) if abs(c) > 1e-9 else float("inf"))
        here = math.hypot(cx, cy)
        margin_mm = d - here
        rows.append({"slew_deg": round(math.degrees(psi), 1),
                     "cog_out_mm": round(here, 1),
                     "edge_mm": round(d, 1),
                     "margin_mm": round(margin_mm, 1),
                     "tip_g": round(margin_mm / float(cog_height_mm), 3)})

    worst = min(rows, key=lambda x: x["tip_g"])
    best = max(rows, key=lambda x: x["tip_g"])
    static = (float(track_mm) / 2.0) / float(cog_height_mm)

    out = {
        "static_tip_g": round(static, 3),
        "worst_tip_g": worst["tip_g"], "worst_slew_deg": worst["slew_deg"],
        "best_tip_g": best["tip_g"], "best_slew_deg": best["slew_deg"],
        "cog_offset_mm": round(offset, 1), "total_kg": round(total, 3),
        "curve": rows, "warnings": [], "notes": [],
    }
    if worst["tip_g"] <= 0:
        out["warnings"].append(
            f"IT TIPS OVER ON ITS OWN at {worst['slew_deg']:g} deg of slew: the centre of gravity is "
            f"{worst['cog_out_mm']:.0f} mm out and the wheels end at {worst['edge_mm']:.0f} mm. No "
            f"acceleration needed. Shorten the reach, add counterweight, or widen the track.")
    elif worst["tip_g"] < static * 0.5:
        out["warnings"].append(
            f"reaching out costs more than half the standing stability: {static:.2f} g parked against "
            f"{worst['tip_g']:.2f} g at {worst['slew_deg']:g} deg of slew. The static number flatters "
            f"this machine and should not be quoted on its own.")
    out["notes"].append(
        f"worst heading is {worst['slew_deg']:g} deg ({worst['tip_g']:.2f} g), best is "
        f"{best['slew_deg']:g} deg ({best['tip_g']:.2f} g)")
    return out


def drive(*, mass_kg: float, wheel_dia_mm: float, crr: float = 0.03, mu: float = 0.9,
          gear_ratio: float = 1.0, motor_stall_Nm: float = 0.0, driven_wheels: int = 4,
          grade_deg: float = 10.0, accel_ms2: float = 0.3, cog_height_mm: float = 0.0,
          wheelbase_mm: float = 0.0) -> dict:
    """What it takes to MOVE the thing, and what stops it first.

    Three demands add: rolling resistance, climbing a grade, and accelerating. The answer is a
    torque per motor, but the number that actually matters is which limit binds. A drive can be
    torque-limited (the motor stalls) or traction-limited (the wheels spin), and they want opposite
    fixes -- more reduction helps the first and does nothing at all for the second.

    Grade is the design case here rather than flat ground: a machine that only just moves on the
    level cannot climb the lip of a tray, and that is the failure you meet first on a desk.
    """
    if mass_kg <= 0 or wheel_dia_mm <= 0:
        raise ValueError("mass and wheel diameter must be positive")
    g, r = 9.81, float(wheel_dia_mm) / 2000.0            # metres
    w = float(mass_kg) * g
    f_roll = float(crr) * w
    f_grade = w * math.sin(math.radians(float(grade_deg)))
    f_accel = float(mass_kg) * float(accel_ms2)
    f_need = f_roll + f_grade + f_accel

    # Traction: on a grade the normal force falls with cos, so the limit falls with it too.
    f_trac = float(mu) * w * math.cos(math.radians(float(grade_deg)))
    t_wheel = f_need * r                                  # N.m at the axle, whole vehicle
    n = max(1, int(driven_wheels))
    t_motor = t_wheel / (n * max(1e-9, float(gear_ratio)))
    have = float(motor_stall_Nm or 0.0) * float(gear_ratio) * n / max(1e-9, r)   # N of push available

    warnings, limited_by = [], "nothing computed"
    if motor_stall_Nm:
        limited_by = "traction" if have > f_trac else "motor torque"
        if have < f_need:
            warnings.append(
                f"the motors make {have:.1f} N of push and moving it needs {f_need:.1f} N on a "
                f"{grade_deg:g} degree grade — it will not climb")
        elif limited_by == "traction":
            warnings.append(
                f"traction-limited, not torque-limited: {have:.0f} N available against a "
                f"{f_trac:.1f} N friction ceiling, so the wheels spin before the motors stall. "
                f"More reduction buys nothing here; grip or weight does")

    # Pitching forward under braking is the same sum as tipping sideways, about the front axle.
    tip_accel = None
    if cog_height_mm > 0 and wheelbase_mm > 0:
        tip_accel = g * (float(wheelbase_mm) / 2.0) / float(cog_height_mm)
        if tip_accel < float(mu) * g:
            warnings.append(
                f"it pitches before it slides: {tip_accel:.1f} m/s2 tips it over the front axle and "
                f"the tyres hold to {mu * g:.1f} m/s2. Braking hard is the risk, not spinning up")
    return {"force_N": {"rolling": round(f_roll, 2), "grade": round(f_grade, 2),
                        "accel": round(f_accel, 2), "total": round(f_need, 2)},
            "torque_at_wheel_Nm": round(t_wheel, 3),
            "torque_per_motor_Nm": round(t_motor, 4),
            "motor_push_N": round(have, 1) if motor_stall_Nm else None,
            "traction_ceiling_N": round(f_trac, 1),
            "limited_by": limited_by,
            "margin_x": round(have / f_need, 1) if (motor_stall_Nm and f_need > 0) else None,
            "tip_forward_at_ms2": None if tip_accel is None else round(tip_accel, 2),
            "assumes": {"grade_deg": grade_deg, "accel_ms2": accel_ms2, "crr": crr, "mu": mu,
                        "driven_wheels": n, "gear_ratio": gear_ratio},
            "warnings": warnings}


def gear_mesh(module_mm: float, teeth_a: int, teeth_b: int, *, internal: bool = False,
              placed_centre_mm: float | None = None, tol_mm: float = 0.05) -> dict:
    """Where two gears have to sit to actually touch, and whether they were put there.

    External: (Da + Db)/2. Internal: (Dring - Dpinion)/2, which is the one people get wrong -- it is
    a DIFFERENCE, not a sum, and using the external formula on an internal pair puts the pinion
    outside the ring entirely.

    The slew pair on this arm was placed at 45 mm when it needed 40.5. Every individual number was
    right: right module, right tooth counts, right ratio, gears that both printed and both measured
    correctly. They simply did not touch, and nothing in the pipeline asked whether they would --
    the same shape of failure as the planetary, one level up.
    """
    m = float(module_mm)
    da, db = m * int(teeth_a), m * int(teeth_b)
    need = (abs(da - db) if internal else (da + db)) / 2.0
    out = {"module_mm": m, "teeth": [int(teeth_a), int(teeth_b)], "internal": bool(internal),
           "pitch_dia_mm": [da, db], "centre_distance_mm": round(need, 3),
           "ratio": round(max(da, db) / min(da, db), 3), "warnings": []}
    if internal and min(int(teeth_a), int(teeth_b)) >= max(int(teeth_a), int(teeth_b)):
        out["warnings"].append("an internal pair needs the ring to have more teeth than the pinion")
    if placed_centre_mm is not None:
        err = float(placed_centre_mm) - need
        out["placed_mm"] = float(placed_centre_mm)
        out["error_mm"] = round(err, 3)
        out["meshes"] = abs(err) <= float(tol_mm)
        if not out["meshes"]:
            out["warnings"].append(
                f"THE GEARS DO NOT MESH: placed {placed_centre_mm:g} mm apart, needs {need:g} "
                f"({err:+.1f} mm out). "
                + ("Too far and the teeth never touch." if err > 0 else
                   "Too close and they jam or interfere.")
                + (" An INTERNAL pair is (Dring - Dpinion)/2, a difference, not a sum -- using the "
                   "external formula is how a pinion ends up outside its own ring."
                   if internal else ""))
    return out


def _selfcheck() -> None:
    # ── the ram's moment arm is not constant, and dies at the toggle ──
    r = ram(40, 40, 5, 175, force_N=480)
    assert r["moment_arm_min_mm"] < 0.1 * r["moment_arm_max_mm"]
    assert any("TOGGLE" in w for w in r["warnings"])

    # A sane range well away from both toggles has a live moment arm throughout.
    ok = ram(40, 40, 55, 150, force_N=480, load_torque_Nm=1.7)
    assert not any("TOGGLE" in w for w in ok["warnings"]), ok["warnings"]
    assert ok["margin_min"] > 1.0, ok["margin_min"]
    assert ok["stroke_mm"] > 0

    # The worst pose is a real position inside the sweep, not an assumption.
    assert ok["range_deg"][0] <= ok["margin_worst_angle_deg"] <= ok["range_deg"][1]

    # ── workspace: the elbow limit really does shorten the reach ──
    w = workspace(180, 140, joint2_deg=120)
    assert w["reach_mm"] < w["reach_if_straight_mm"]
    assert workspace(180, 140, joint2_deg=180)["reach_mm"] > w["reach_mm"]

    # ── tip-over: stowed, it must reproduce the old static formula exactly ──
    stow = tipover(track_mm=180, wheelbase_mm=220, cog_height_mm=55,
                   base_kg=2.6, arm_kg=1.23, payload_kg=0, reach_mm=0)
    assert abs(stow["worst_tip_g"] - stow["static_tip_g"]) < 0.02, (stow["worst_tip_g"],
                                                                    stow["static_tip_g"])
    # ...and extended, it must be worse. If it is not, the model is doing nothing.
    out = tipover(track_mm=180, wheelbase_mm=220, cog_height_mm=55,
                  base_kg=2.6, arm_kg=1.23, payload_kg=0.5, reach_mm=320)
    assert out["worst_tip_g"] < stow["worst_tip_g"], (out["worst_tip_g"], stow["worst_tip_g"])

    # ── gears: the slew pair, at the distance the build type now places it ──
    g = gear_mesh(1.5, 72, 18, internal=True, placed_centre_mm=40.5)
    assert g["centre_distance_mm"] == 40.5 and g["meshes"] is True, g
    bad = gear_mesh(1.5, 72, 18, internal=True, placed_centre_mm=45.0)
    assert bad["meshes"] is False and any("DO NOT MESH" in w for w in bad["warnings"])
    # the external formula on an internal pair is the classic error, and must not silently pass
    assert gear_mesh(1.5, 72, 18, internal=False)["centre_distance_mm"] == 67.5

    print(f"mechanism ok — ram worst arm {ok['moment_arm_min_mm']:.1f} mm at "
          f"{ok['margin_worst_angle_deg']:g} deg (margin {ok['margin_min']}x); "
          f"reach {w['reach_mm']:.0f} of {w['reach_if_straight_mm']:.0f} mm; "
          f"tip-over {stow['static_tip_g']:.2f} g parked -> {out['worst_tip_g']:.2f} g at "
          f"{out['worst_slew_deg']:g} deg")


if __name__ == "__main__":
    _selfcheck()
