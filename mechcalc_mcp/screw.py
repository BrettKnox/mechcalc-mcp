"""screw.py — power-screw mechanics from geometry and friction, instead of from an assumption.

An arm built on leadscrews rests on one claim: that the screw is self-locking, so the boom holds its
position with the power off. Nothing in this codebase checked it. `engines/linear.py` decided it with
`holding = force if efficiency <= 0.5`, keyed on an efficiency the designer had typed in — so the
conclusion "it self-locks" was derived from the assumption "eta = 0.35", which was itself chosen
because the screw was assumed to self-lock. A circle.

Breaking it needs one number the design never wrote down: the LEAD ANGLE. A screw self-locks when its
lead angle is shallower than its friction angle, and that is decided by lead and diameter, not by
anybody's opinion of efficiency. For T8 the two candidate leads land either side of the line:

    T8 2 mm lead   lambda =  5.2 deg   needs mu >= 0.088   -> locks on any plausible friction
    T8 4 mm lead   lambda = 10.3 deg   needs mu >= 0.176   -> does NOT lock once greased

The 4 mm screw was the chosen one. Worse, eta = 0.35 is the 2 mm screw's efficiency (0.364 at
mu = 0.15); the 4 mm runs at about 0.52. So the design was using 2 mm physics on a 4 mm screw, and
eta > 0.5 is itself the definition of not self-locking.

Friction is a RANGE, never a number, so every verdict here is computed across a band and reports the
worst end. A design that self-locks at the optimistic end of a friction range does not self-lock.
"""
from __future__ import annotations

import math

# Trapezoidal metric thread, ISO 2904. Half-angle of the 30-degree included form.
TRAPEZOIDAL_HALF_ANGLE_DEG = 15.0
SQUARE_HALF_ANGLE_DEG = 0.0

# Steel screw against its nut. Ranges, because a single number here is the whole problem this module
# exists to fix: pick the optimistic end and everything self-locks on paper.
# Every band is (low, high) and the SELF-LOCKING verdict is taken at the LOW end, because the
# lowest credible friction is the one that lets a loaded screw back-drive. A nominal value is the
# wrong number to hold a boom up with.
#
# `steel_brass_graphite` is here to be selectable, not because it is recommended: graphite and
# EP/moly additives roughly halve brass friction (Bhandari lists 0.04-0.06 for oil-with-graphite),
# and at 0.04 a 2 mm lead T8 -- which needs 0.088 -- does NOT self-lock. The grease is therefore a
# DESIGN REQUIREMENT of this build and not a detail left to whatever is on the bench.
FRICTION = {
    "steel_brass_dry":     (0.15, 0.19),
    "steel_brass_greased": (0.10, 0.15),
    "steel_brass_graphite": (0.04, 0.06),   # graphite/EP grease: BREAKS self-locking on a fine lead
    "steel_bronze_dry":    (0.15, 0.19),
    "steel_pom_dry":       (0.10, 0.20),   # a printed or POM anti-backlash nut
    "steel_steel_dry":     (0.15, 0.25),
}
DEFAULT_FRICTION = "steel_brass_greased"   # what a leadscrew in service actually is

# Standard Tr8 geometry (d = 8, P = 2): pitch dia 7.0, minor dia 5.5.
THREADS = {
    "T8": {"major_mm": 8.0, "pitch_mm": 2.0, "pitch_dia_mm": 7.0, "minor_dia_mm": 5.5},
    "T10": {"major_mm": 10.0, "pitch_mm": 2.0, "pitch_dia_mm": 9.0, "minor_dia_mm": 7.5},
    "T12": {"major_mm": 12.0, "pitch_mm": 3.0, "pitch_dia_mm": 10.5, "minor_dia_mm": 8.5},
}

# Buckling end-fixity (Euler K) and critical-speed end-fixity (Cs). Different tables, different
# physics: one is about a column folding, the other about a shaft whipping.
BUCKLING_K = {"pinned": 1.0, "fixed_free": 2.0, "fixed_fixed": 0.5, "fixed_pinned": 0.7}
WHIP_CS = {"pinned": 1.0, "fixed_free": 0.36, "fixed_fixed": 2.23, "fixed_pinned": 1.47}

# Metric critical-speed constant. The commonly quoted 4.76e6 is per INCH; x25.4 for mm.
WHIP_K_MM = 1.21e8

STEEL_E_MPA = 200_000.0
STEEL_YIELD_MPA = 235.0                    # plain carbon leadscrew stock, not a bolt grade


def _geom(thread: str, lead_mm: float | None = None, starts: int = 1) -> dict:
    t = THREADS.get(str(thread).upper())
    if not t:
        raise ValueError(f"{thread!r} is not a thread this knows; try {', '.join(THREADS)}")
    lead = float(lead_mm) if lead_mm else t["pitch_mm"] * max(1, int(starts))
    return {**t, "lead_mm": lead}


def analyse(thread: str = "T8", lead_mm: float = 2.0, *, friction: str = DEFAULT_FRICTION,
            half_angle_deg: float = TRAPEZOIDAL_HALF_ANGLE_DEG,
            load_N: float = 0.0, motor_torque_Nm: float = 0.0,
            free_length_mm: float = 0.0, end: str = "pinned") -> dict:
    """Everything a leadscrew choice decides, from its geometry and a friction BAND.

    `lead` is travel per revolution and is what sets the mechanics; pitch is thread-to-thread and is
    the same 2 mm for every T8 whatever its start count, which is exactly how a 4-start screw gets
    mistaken for a 1-start one in a catalogue.
    """
    g = _geom(thread, lead_mm)
    lead = g["lead_mm"]
    d_m = g["pitch_dia_mm"] / 1000.0                      # metres, for torque
    d_r = g["minor_dia_mm"]

    lo, hi = FRICTION.get(friction, FRICTION[DEFAULT_FRICTION])
    ca = math.cos(math.radians(half_angle_deg))
    tan_lambda = lead / (math.pi * g["pitch_dia_mm"])
    lam_deg = math.degrees(math.atan(tan_lambda))

    def eff(mu):
        return (ca - mu * tan_lambda) / (ca + mu / tan_lambda) if tan_lambda else 0.0

    # Self-locking: mu >= tan(lambda) * cos(alpha). Judged at the LOW end of the friction band,
    # because the slipperiest plausible screw is the one that has to hold.
    mu_needed = tan_lambda * ca
    locks_worst = lo >= mu_needed
    locks_best = hi >= mu_needed
    # A screw that only just locks statically still walks down under vibration, and this arm rides
    # on a driven chassis. The usual guard is a healthy factor on the criterion, not equality.
    locks_vibration = lo >= mu_needed * 1.5

    eta_lo, eta_hi = eff(hi), eff(lo)                     # more friction, less efficiency

    out = {
        "thread": thread, "lead_mm": lead, "pitch_mm": g["pitch_mm"],
        "starts": round(lead / g["pitch_mm"], 2),
        "pitch_dia_mm": g["pitch_dia_mm"], "minor_dia_mm": d_r,
        "lead_angle_deg": round(lam_deg, 2),
        "friction": friction, "mu_range": [lo, hi],
        "mu_needed_to_lock": round(mu_needed, 4),
        "self_locking": bool(locks_worst),
        "self_locking_at_best_friction_only": bool(locks_best and not locks_worst),
        "self_locking_under_vibration": bool(locks_vibration),
        "efficiency": round(eta_lo, 3),                   # the pessimistic end, for sizing force
        "efficiency_range": [round(eta_lo, 3), round(eta_hi, 3)],
        "warnings": [], "notes": [],
    }

    if not locks_worst:
        out["warnings"].append(
            f"NOT SELF-LOCKING: a {lead:g} mm lead on {thread} is a {lam_deg:.1f} deg lead angle and "
            f"needs mu >= {mu_needed:.3f} to hold. {friction} is {lo}-{hi}, so at the slippery end it "
            f"back-drives. The arm falls when the power is cut."
            + (" It would hold at the high-friction end, which is not a design." if locks_best else ""))
    elif not locks_vibration:
        out["warnings"].append(
            f"self-locks statically ({mu_needed:.3f} needed against {lo}-{hi}) but with less than 1.5x "
            f"margin. On a moving vehicle a marginal screw walks down over time; treat this as "
            f"needing a detent or a finer lead.")
    else:
        out["notes"].append(
            f"self-locking with margin: needs mu >= {mu_needed:.3f}, has {lo}-{hi}")

    if eta_lo > 0.5:
        out["warnings"].append(
            f"efficiency {eta_lo:.2f} is above 0.5, which is the same statement as 'not self-locking' "
            f"seen from the other side. The two cannot disagree, so a design claiming both is wrong "
            f"about one of them.")

    # ── force and torque ──
    if motor_torque_Nm:
        out["force_N"] = round(2 * math.pi * motor_torque_Nm * eta_lo / (lead / 1000.0), 1)
        out["force_range_N"] = [
            round(2 * math.pi * motor_torque_Nm * eta_lo / (lead / 1000.0), 1),
            round(2 * math.pi * motor_torque_Nm * eta_hi / (lead / 1000.0), 1)]
    if load_N:
        mu = lo
        raise_t = load_N * d_m / 2 * (ca * tan_lambda + mu) / (ca - mu * tan_lambda)
        lower_t = load_N * d_m / 2 * (mu - ca * tan_lambda) / (ca + mu * tan_lambda)
        out["raise_torque_Nm"] = round(raise_t, 4)
        out["backdrive_torque_Nm"] = round(lower_t, 4)
        out["notes"].append(
            f"holding {load_N:g} N needs {raise_t:.3f} N.m to raise; "
            + (f"the load itself would drive it back with {abs(lower_t):.3f} N.m if nothing resisted"
               if lower_t < 0 else
               f"{lower_t:.3f} N.m is needed to lower it, which is what self-locking means"))

    # ── the screw as a column, and as a spinning shaft ──
    if free_length_mm and free_length_mm > 0:
        L = float(free_length_mm)
        I = math.pi * (d_r ** 4) / 64.0                       # mm^4, on the MINOR diameter
        K = BUCKLING_K.get(end, 1.0)
        p_cr = (math.pi ** 2) * STEEL_E_MPA * I / ((K * L) ** 2)          # N
        area = math.pi * (d_r ** 2) / 4.0
        p_yield = STEEL_YIELD_MPA * area
        out["buckling_load_N"] = round(p_cr, 1)
        out["yield_load_N"] = round(p_yield, 1)
        out["compression_limit_N"] = round(min(p_cr, p_yield), 1)
        out["limited_by"] = "buckling" if p_cr < p_yield else "yield"
        if load_N and min(p_cr, p_yield) < load_N * 2:
            out["warnings"].append(
                f"the screw itself is the weak link in compression: {min(p_cr, p_yield):.0f} N "
                f"({out['limited_by']}) against a {load_N:g} N load over {L:g} mm free length. "
                f"Shorten the unsupported span or fix the far end.")

        # N_c = C * K * d_root / L^2. The constant published for this formula, 4.76e6, is for INCHES,
        # and feeding it millimetres understates the answer by 25.4x -- which is exactly what this
        # module did at first, and it produced a confident "whips at 164 rpm, inside your operating
        # range" that was simply not true. Checked against the first bending mode of a
        # simply-supported steel shaft, f1 = (pi/2)*sqrt(EI/(rho*A*L^4)): a 5.5 mm root over 400 mm
        # is 68 Hz, i.e. about 4090 rpm, and the metric constant below reproduces it.
        cs = WHIP_CS.get(end, 1.0)
        n_c = WHIP_K_MM * d_r * cs / (L ** 2)
        out["critical_speed_rpm"] = round(n_c, 0)
        out["safe_speed_rpm"] = round(n_c * 0.8, 0)
        out["notes"].append(
            f"whip starts around {n_c:.0f} rpm over {L:g} mm ({end}); stay under "
            f"{n_c * 0.8:.0f} rpm")
    return out


def compare(thread: str = "T8", leads=(2.0, 4.0, 8.0), **kw) -> list[dict]:
    """The same screw at several leads, which is the choice actually being made."""
    return [analyse(thread, lead_mm=lead, **kw) for lead in leads]


def _selfcheck() -> None:
    # The headline: the two candidate leads fall on opposite sides of the self-locking line, and
    # NOTHING in the design could see that before this module existed.
    two = analyse("T8", 2.0, friction="steel_brass_greased")
    four = analyse("T8", 4.0, friction="steel_brass_greased")

    assert 5.1 < two["lead_angle_deg"] < 5.3, two["lead_angle_deg"]
    assert 10.2 < four["lead_angle_deg"] < 10.4, four["lead_angle_deg"]

    assert two["self_locking"] is True, two
    assert four["self_locking"] is False, four
    assert any("NOT SELF-LOCKING" in w for w in four["warnings"])

    # eta = 0.35 was the assumption the whole design used. It is the 2 mm screw's number.
    assert 0.35 < two["efficiency"] < 0.38, two["efficiency"]
    assert four["efficiency"] > 0.5, four["efficiency"]
    # ...and above 0.5 the module must say the two claims contradict each other.
    assert any("cannot disagree" in w for w in four["warnings"])

    # Efficiency and lead trade: the FINER screw makes more force from the same motor.
    f2 = analyse("T8", 2.0, motor_torque_Nm=0.42)["force_N"]
    f4 = analyse("T8", 4.0, motor_torque_Nm=0.42)["force_N"]
    assert f2 > f4, (f2, f4)

    # Friction is a band and the verdict must be taken at the slippery end.
    dry = analyse("T8", 4.0, friction="steel_brass_dry")
    assert dry["self_locking"] is False
    assert dry["self_locking_at_best_friction_only"] is True, dry

    # The screw as a column and as a shaft. Cutting it to length is worth four times the margin:
    # the as-bought stick buckles BELOW the force the motor can make, and whips inside our rpm range.
    uncut = analyse("T8", 2.0, load_N=480, free_length_mm=400, end="pinned")
    cut = analyse("T8", 2.0, load_N=480, free_length_mm=span_for(64), end="pinned")
    assert uncut["limited_by"] == "buckling" and uncut["compression_limit_N"] < 480 * 1.5
    assert any("weak link in compression" in w for w in uncut["warnings"])
    assert cut["limited_by"] == "yield" and cut["compression_limit_N"] > 480 * 5
    assert cut["warnings"] == [] or not any("weak link" in w for w in cut["warnings"])
    # Whip: checked against the first bending mode of a simply-supported shaft, because the
    # published 4.76e6 constant is per INCH and feeding it mm understated this by 25.4x.
    _E, _rho, _d, _L = 200e9, 7850.0, 0.0055, 0.400
    _I, _A = math.pi * _d ** 4 / 64, math.pi * _d ** 2 / 4
    _rpm = (math.pi / 2) * math.sqrt(_E * _I / (_rho * _A * _L ** 4)) * 60
    assert abs(uncut["critical_speed_rpm"] - _rpm) / _rpm < 0.05, uncut["critical_speed_rpm"]
    assert uncut["critical_speed_rpm"] > 3000, "whip is not a problem on this machine"

    # Backdrive: a self-locking screw needs torque to LOWER it; a non-locking one gives it back.
    a = analyse("T8", 2.0, load_N=300)
    b = analyse("T8", 4.0, load_N=300)
    assert a["backdrive_torque_Nm"] > 0, a["backdrive_torque_Nm"]
    assert b["backdrive_torque_Nm"] < 0, b["backdrive_torque_Nm"]

    print(f"screw ok — T8 2mm: {two['lead_angle_deg']} deg, eta {two['efficiency']}, "
          f"self-locking {two['self_locking']}; T8 4mm: {four['lead_angle_deg']} deg, "
          f"eta {four['efficiency']}, self-locking {four['self_locking']}")


def span_for(stroke_mm: float, nut_mm: float = 15.0, fixings_mm: float = 30.0) -> float:
    """The span that actually matters: what is UNSUPPORTED between the end bearings at full stroke.

    Not the length of screw you buy. A T8 is sold in 300-500 mm sticks and this arm's rams stroke
    under 90 mm, so a builder who bolts in the whole stick leaves a 400 mm free span -- which BUCKLES
    at 554 N, below the 480 N the motor can push. Cut to length and the same screw yields at 5583 N
    instead: same part, ten times the margin, one saw cut.

    Whip is NOT a reason to cut it. An earlier version of this claimed the uncut screw whirled at
    164 rpm, which was the imperial constant applied to millimetres; the real figure is about
    4100 rpm and nothing on this machine goes near it. Buckling alone justifies the cut.
    """
    return float(stroke_mm) + float(nut_mm) + float(fixings_mm)


if __name__ == "__main__":
    print(f"as-bought 400 mm stick, uncut, 480 N: "
          f"{analyse('T8', 2.0, load_N=480, free_length_mm=400)['compression_limit_N']:.0f} N limit, "
          f"{analyse('T8', 2.0, free_length_mm=400)['critical_speed_rpm']:.0f} rpm whip  <- CUT IT")
    cut = span_for(64)
    print(f"cut to {cut:.0f} mm for a 64 mm stroke:  "
          f"{analyse('T8', 2.0, load_N=480, free_length_mm=cut)['compression_limit_N']:.0f} N limit, "
          f"{analyse('T8', 2.0, free_length_mm=cut)['critical_speed_rpm']:.0f} rpm whip")

    for r in compare("T8", (2.0, 4.0, 8.0), motor_torque_Nm=0.42, load_N=300,
                     free_length_mm=span_for(64), end="pinned"):
        print(f"\nT8 lead {r['lead_mm']:g} mm  ({r['starts']:g}-start)")
        print(f"   lead angle {r['lead_angle_deg']:5.2f} deg   needs mu >= {r['mu_needed_to_lock']:.3f}"
              f"   has {r['mu_range']}")
        print(f"   self-locking {str(r['self_locking']):5}  under vibration "
              f"{str(r['self_locking_under_vibration']):5}  eta {r['efficiency_range']}")
        print(f"   force {r.get('force_N')} N   backdrive {r.get('backdrive_torque_Nm')} N.m")
        print(f"   compression limit {r.get('compression_limit_N')} N ({r.get('limited_by')})   "
              f"whip {r.get('critical_speed_rpm')} rpm")
        for w in r["warnings"]:
            print(f"   ! {w}")
    print()
    _selfcheck()
