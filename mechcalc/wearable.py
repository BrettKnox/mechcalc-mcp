"""wearable.py — the two checks a body-worn device lives or dies by, deterministic and inline.

Neither of these existed, so an orthosis got designed with structural tools only, and the two things
that actually decide whether a brace works were left to judgement:

  1. INTERFACE PRESSURE. A brace transmits its load through skin. Skin has a budget, not a limit that
     scales with how hard you pull a strap. p = F/A, and the only real lever is AREA. Thresholds used
     here: <=20 mmHg tolerated for prolonged wear, 32 mmHg the arteriole-occlusion figure commonly
     quoted, 33 mmHg proposed as the ceiling beyond 8 hours. These are a design budget plus a skin
     check, not a guarantee -- they shift with limb elevation, temperature and general health.

  2. ROM COMPLIANCE. A joint stop does NOT deliver its labelled angle. Measured on a Bauerfeind SofTec
     Coxa: set 70 deg -> 89.4 deg achieved; set 90 deg -> 99.2 deg achieved; unhinged 107.4 deg. The
     gap is joint flex under load plus residual motion between brace and limb. So the stop must be set
     BELOW the target by the slip, and the slip is a per-wearer measurement, not a constant.

Units: N, cm^2, mmHg, kPa, degrees. No LLM, no compute job -- microseconds of arithmetic.
"""
from __future__ import annotations

import math

MMHG_PER_KPA = 7.50062                  # 1 kPa = 7.50062 mmHg
PRESSURE_PROLONGED_MMHG = 20.0          # tolerated for prolonged wear
PRESSURE_OCCLUSION_MMHG = 32.0          # commonly quoted arteriole occlusion
PRESSURE_8H_CEILING_MMHG = 33.0         # proposed ceiling for wear beyond 8 h

# Not all skin is equal, and treating it as if it were is what makes a brace either useless or
# injurious. A cuff presses on muscle and fat that were never meant to carry load. The ischial
# tuberosity carries a seated adult all day: measured seating pressures under it run 40-60 mmHg on a
# pressure-relieving cushion and 95-120 mmHg on a bare wheelchair sling. So the seat gets a higher
# budget than the cuffs -- but only while the wearer is SENSATE and can shift; those seating figures
# come from spinal-cord-injury populations where the whole problem is that they cannot feel it.
SITES = {
    "cuff":   {"prolonged": 20.0, "ceiling": 32.0,
               "note": "soft tissue over muscle/fat: stabilising forces only, not a load path"},
    "seat":   {"prolonged": 60.0, "ceiling": 100.0,
               "note": "ischial tuberosity, intermittent bearing; requires an offload every 20-30 min"},
    "bony":   {"prolonged": 0.0, "ceiling": 0.0,
               "note": "bony prominence: relieve entirely, any bearing here is a pressure injury"},
}

# Measured slip on a commercial hip brace: (set angle -> achieved angle).
HIP_BRACE_SLIP_DEG = 19.4               # mean of (89.4-70) and (99.2-90)
BONY_PROMINENCES = {
    "hip": ["ASIS", "greater trochanter", "iliac crest", "ischial tuberosity (seat: intended, pad it)"],
    "knee": ["fibular head (common peroneal nerve)", "patella", "tibial tubercle"],
    "shank": ["tibial crest", "medial malleolus", "lateral malleolus"],
}


# A body-worn load is never the standing number. Ground reaction rises with what the wearer is doing,
# and the interface sees all of it. Sizing a pad on the standing load is how a design that "passes"
# spends every step over budget: our ischial seat was 51.7 mmHg standing and 93.1 mmHg on stair
# descent, within 7% of the ceiling, and nothing in the tool asked the question.
GAIT_FACTOR = {
    "standing": 1.0,
    "walking": 1.2,
    "walking_peak": 1.5,
    "stairs": 1.8,
}


def dynamic_check(force_n: float, pad_area_cm2: float, site: str = "cuff",
                  hours: float = 8.0, label: str = "pad") -> dict:
    """The same pad across every gait condition, because standing is the one load it never sees.

    Returns a row per condition and the worst verdict, so a pad is sized for the wearer's day rather
    than for the moment they stand still on a fitting stool.
    """
    rows, worst = [], "pass"
    for name, k in GAIT_FACTOR.items():
        r = interface_pressure(force_n * k, pad_area_cm2, hours, f"{label}/{name}", site)
        if not r.get("ok"):
            return r
        rows.append({"condition": name, "factor": k, "force_N": r["force_N"],
                     "pressure_mmHg": r["pressure_mmHg"], "verdict": r["verdict"]})
        if r["verdict"] == "FAIL":
            worst = "FAIL"
        elif r["verdict"] == "marginal" and worst == "pass":
            worst = "marginal"
    need = interface_pressure(force_n * max(GAIT_FACTOR.values()), pad_area_cm2, hours, label, site)
    return {"ok": True, "label": label, "site": site, "static_force_N": round(float(force_n), 1),
            "pad_area_cm2": round(float(pad_area_cm2), 1), "conditions": rows,
            "worst_verdict": worst, "passes_everywhere": worst == "pass",
            "area_for_worst_case_cm2": need["area_needed_cm2"]["prolonged"],
            "note": ("sized for the whole day" if worst == "pass" else
                     f"worst condition is {worst}: this pad is outside the sustained budget somewhere "
                     f"in normal use. Size it for the hardest thing the wearer will actually do, not "
                     f"for standing — or restrict that activity and say so.")}


def load_capacity(pad_area_cm2: float, site: str = "cuff", sustained: bool = True) -> dict:
    """The other direction, and the one that actually sizes a brace: given the area you can realistically
    get onto the body at this site, how much force may pass through it?

    Run this BEFORE designing a load path. A thigh cuff at 300 cm^2 carries 80 N at the prolonged
    budget -- not the several hundred newtons an unloading claim usually assumes.
    """
    try:
        a = float(pad_area_cm2)
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": f"bad numeric input: {e}"}
    if a <= 0:
        return {"ok": False, "error": "pad_area_cm2 must be > 0"}
    b = SITES.get(site)
    if b is None:
        return {"ok": False, "error": f"unknown site '{site}'", "known": sorted(SITES)}
    limit = b["prolonged"] if sustained else b["ceiling"]
    if limit <= 0:
        return {"ok": True, "site": site, "pad_area_cm2": a, "max_force_N": 0.0,
                "limit_mmHg": 0.0, "note": b["note"]}
    return {"ok": True, "site": site, "pad_area_cm2": round(a, 1),
            "basis": "sustained" if sustained else "short-duration ceiling",
            "limit_mmHg": limit, "max_force_N": round(limit / MMHG_PER_KPA * 1000.0 * a * 1e-4, 1),
            "note": b["note"]}


def interface_pressure(force_n: float, pad_area_cm2: float, hours: float = 8.0,
                       label: str = "cuff", site: str = "cuff") -> dict:
    """Pressure a cuff/strap puts into skin, against the wear-time budget.

    Returns the pressure, the verdict, and -- the useful part -- the pad area that WOULD pass, since
    area is the only lever that does not also hurt.
    """
    try:
        f, a, h = float(force_n), float(pad_area_cm2), float(hours)
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": f"bad numeric input: {e}"}
    if a <= 0:
        return {"ok": False, "error": "pad_area_cm2 must be > 0"}
    if f < 0:
        return {"ok": False, "error": "force_n must be >= 0"}

    b = SITES.get(site)
    if b is None:
        return {"ok": False, "error": f"unknown site '{site}'", "known": sorted(SITES)}
    kpa = (f / (a * 1e-4)) / 1000.0                     # N/m^2 -> kPa
    mmhg = kpa * MMHG_PER_KPA
    prolonged, ceiling = b["prolonged"], b["ceiling"]
    # Long wear is judged against the prolonged budget; a short bout may use the ceiling.
    limit = prolonged if h > 8 else ceiling
    if mmhg <= prolonged:
        verdict, ok = "pass", True
    elif mmhg <= limit:
        verdict, ok = "marginal", True
    else:
        verdict, ok = "FAIL", False

    def area_for(target_mmhg: float) -> float:
        if target_mmhg <= 0:
            return float("inf")
        return f / (target_mmhg / MMHG_PER_KPA * 1000.0) * 1e4    # cm^2

    out = {
        "ok": True, "label": label, "site": site, "force_N": round(f, 1),
        "pad_area_cm2": round(a, 1),
        "hours": h, "pressure_mmHg": round(mmhg, 1), "pressure_kPa": round(kpa, 2),
        "verdict": verdict, "passes": ok,
        "limit_applied_mmHg": limit, "site_note": b["note"],
        "area_needed_cm2": {"prolonged": round(area_for(prolonged), 1),
                            "ceiling": round(area_for(ceiling), 1)},
        "notes": [
            "p = F/A: widen and lengthen the pad, do not tighten the strap",
            "these thresholds move with limb elevation, temperature and general health -- "
            "budget plus a skin check at 20 min and 2 h, not a guarantee",
            "pressure over a bony prominence concentrates far above this average -- relieve it",
        ],
    }
    if not ok:
        need = out["area_needed_cm2"]["prolonged"]
        out["fix"] = (f"pad area {a:.1f} cm^2 -> at least {need:.1f} cm^2 "
                      f"({need / a:.1f}x), or cut the force carried here")
    return out


def rom_calibrate(target_deg: float, slip_deg: float = HIP_BRACE_SLIP_DEG,
                  step_deg: float = 5.0, joint: str = "hip") -> dict:
    """What to SET a joint stop to so the wearer actually achieves target_deg.

    slip_deg defaults to the measured hip-brace figure. Measure it on the actual wearer with a
    goniometer and pass it in -- it is the single number that decides whether the limit is real.
    """
    try:
        t, s, step = float(target_deg), float(slip_deg), float(step_deg)
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": f"bad numeric input: {e}"}
    if step <= 0:
        return {"ok": False, "error": "step_deg must be > 0"}

    ideal = t - s
    setting = round(ideal / step) * step                    # snap to a real detent
    if setting > ideal:                                     # never round toward MORE motion
        setting -= step
    achieved = setting + s
    out = {
        "ok": True, "joint": joint, "target_deg": t, "slip_deg": s, "step_deg": step,
        "set_stop_to_deg": setting, "predicted_achieved_deg": round(achieved, 1),
        "margin_deg": round(t - achieved, 1),
        "table": [{"stop_deg": d, "predicted_deg": round(d + s, 1)}
                  for d in _steps(setting, step)],
        "notes": [
            "measured on a commercial hip brace: set 70 -> achieved 89.4; set 90 -> achieved 99.2",
            "slip = joint flex under load + brace-to-limb motion; stiffer coupling and more cuff "
            "area shrink it, a tighter strap mostly does not",
            "re-measure after any weight change or re-fit: slip grows as the cuff loosens",
        ],
    }
    if setting < 0:
        out["ok"] = False
        out["error"] = (f"target {t} deg is below the slip ({s} deg): no stop setting reaches it. "
                        "Stiffen the coupling to cut slip, or accept a higher target.")
    return out


def _steps(centre: float, step: float, n: int = 3) -> list[float]:
    lo = max(0.0, centre - n * step)
    return [lo + i * step for i in range(2 * n + 1)]


# Peak joint flexion each everyday activity actually demands, measured (wearable sensors and
# electrogoniometry), not estimated. This is the table that says what a joint limit COSTS: capping a
# hip at 75 deg is not a mild restriction, it is the difference between walking normally and not being
# able to sit on a chair, use a toilet, or reach your own shoes.
ADL_DEG = [
    # activity,                 hip, knee
    ("walking, level",           49,  75),
    ("stair descent",            52,  90),
    ("stair ascent",             68,  83),
    ("sitting, standard chair",  90,  93),
    ("rising from a chair",     103,  93),
    ("on/off a standard toilet", 113,  93),
    ("getting into a car",      115,  90),
    ("tying shoes",             126, 106),
    ("sitting in a bath",       120, 135),
]


def adl_check(hip_limit_deg: float, knee_limit_deg: float = 90.0) -> dict:
    """Which everyday activities a pair of joint limits allows, and which it takes away.

    A brace specification that states its limits without stating this is only half a specification:
    the wearer finds out the other half the first time they try to sit down.
    """
    try:
        hip, knee = float(hip_limit_deg), float(knee_limit_deg)
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": f"bad numeric input: {e}"}
    rows = []
    for name, h, k in ADL_DEG:
        blocked_by = [w for w, need, lim in (("hip", h, hip), ("knee", k, knee)) if need > lim]
        rows.append({"activity": name, "hip_deg": h, "knee_deg": k,
                     "ok": not blocked_by, "blocked_by": blocked_by})
    lost = [r["activity"] for r in rows if not r["ok"]]
    return {"ok": True, "hip_limit_deg": hip, "knee_limit_deg": knee, "activities": rows,
            "allowed": [r["activity"] for r in rows if r["ok"]], "blocked": lost,
            "note": ("every listed activity is within the limits" if not lost else
                     f"{len(lost)} of {len(rows)} everyday activities are blocked: "
                     f"{', '.join(lost)} — plan equipment for these, do not assume the wearer will manage")}


def offload_gap(target_force_n: float, tissue_k_n_per_mm: float = 20.0,
                adjuster_pitch_mm: float = 1.0, detent_pitch_mm: float = 10.0,
                upright_k_n_per_mm: float = 10000.0) -> dict:
    """How much LONGER than the limb the frame must be to carry `target_force_n`, and what it takes
    to set that dimension.

    A rigid frame beside a leg is a spring in PARALLEL with it. It carries load only because it is
    deliberately a little long: it touches the floor first and must compress before the foot takes
    over. Nothing else makes an unloading brace unload. The spec said "the frame carries 200 N" and
    never said what made it 200 rather than 0 or 600 — this is that missing dimension.

    The compliance is NOT the aluminium. Springs in series are dominated by the softest, and a
    19 × 6 mm upright is ~10,000 N/mm while the tissue under the ischial seat is tens of N/mm. So the
    gap is essentially `target / tissue stiffness`, and **tissue stiffness is a per-wearer
    measurement**: bulk tissue depth under the seated ischium ranges 5.6–32.1 mm between people, so
    no table can supply it. Measure it, or the gap is a guess with three significant figures.

    The same number gives the sensitivity, which is what decides the adjuster: dF/d(gap) = k. At
    20 N/mm a 10 mm detent step moves the load by 200 N — the entire design load, in one click.
    """
    try:
        f = float(target_force_n)
        kt = float(tissue_k_n_per_mm)
        pa = float(adjuster_pitch_mm)
        pd = float(detent_pitch_mm)
        ku = float(upright_k_n_per_mm)
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": f"bad numeric input: {e}"}
    if f <= 0:
        return {"ok": False, "error": "target_force_n must be > 0"}
    if kt <= 0 or ku <= 0:
        return {"ok": False, "error": "stiffnesses must be > 0 N/mm"}

    k = 1.0 / (1.0 / kt + 1.0 / ku)          # series: tissue then structure
    gap = f / k
    step_detent = k * pd
    step_adjuster = k * pa
    quarter_turn = k * pa / 4.0
    return {
        "ok": True, "target_force_N": round(f, 1),
        "tissue_k_N_per_mm": kt, "series_k_N_per_mm": round(k, 2),
        "gap_mm": round(gap, 1),
        "sensitivity_N_per_mm": round(k, 2),
        "load_step_per_detent_N": round(step_detent, 1),
        "load_step_per_adjuster_turn_N": round(step_adjuster, 1),
        "load_step_per_quarter_turn_N": round(quarter_turn, 1),
        "detent_adequate": step_detent <= 0.25 * f,
        "adjuster_adequate": step_adjuster <= 0.25 * f,
        "notes": [
            f"the frame must stand {gap:.1f} mm proud of the limb's own ground contact",
            f"a {pd:g} mm detent moves the load {step_detent:.0f} N — "
            + ("acceptable" if step_detent <= 0.25 * f else
               f"that is {step_detent / f * 100:.0f}% of the whole design load in one click"),
            f"a {pa:g} mm-pitch thread gives {quarter_turn:.0f} N per quarter turn",
            "tissue stiffness is per-wearer and must be measured; the default is a placeholder",
            "the frame's ground contact must be SEPARABLE from the foot's, or this is neither "
            "settable nor measurable",
        ],
    }


def seat_height(hip_limit_deg: float, thigh_len_mm: float = 430.0,
                popliteal_mm: float = 430.0, margin_deg: float = 5.0) -> dict:
    """How high a seat must be for the wearer to sit down without exceeding the hip limit.

    On a standard chair the thigh is horizontal and the hip is at ~90 deg. Raise the seat and the
    thigh slopes down toward the knee, opening the hip angle: flexion = 90 - (thigh declination). So
    the hip must sit `thigh_len * sin(90 - limit)` ABOVE the knee. This is exactly why a raised toilet
    seat is standard equipment under hip precautions — and it is arithmetic, not a preference.

    `margin_deg` aims a little under the limit, because a real sit-down overshoots.
    """
    try:
        lim, lt, pop, m = (float(hip_limit_deg), float(thigh_len_mm), float(popliteal_mm),
                           float(margin_deg))
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": f"bad numeric input: {e}"}
    if not 0 < lim <= 90:
        return {"ok": False, "error": "hip_limit_deg must be between 0 and 90 (above 90 no rise is needed)"}
    target = max(1.0, lim - m)
    decl = math.radians(90.0 - target)
    rise = lt * math.sin(decl)
    return {"ok": True, "hip_limit_deg": lim, "aiming_for_deg": target,
            "thigh_len_mm": lt, "rise_above_knee_mm": round(rise, 1),
            "seat_height_mm": round(pop + rise, 1),
            "standard_seat_mm": round(pop, 1),
            "raise_standard_seat_by_mm": round(rise, 1),
            "notes": [
                "measure thigh length greater trochanter to knee joint centre, on the wearer",
                "a firm seat: a soft cushion sinks and gives the hip flexion back",
                "arms to push up on — rising from a chair normally needs ~103 deg of hip flexion, "
                "which this limit forbids, so the arms do that work instead",
            ]}


def curvature_factor(circ_mm: float, width_mm: float) -> dict:
    """How much a strap's pressure concentrates because the limb is not round.

    Every pressure figure in this module is a MEAN: force over pad area, which assumes the load
    arrives evenly. Under a tensioned strap it does not. Laplace's law gives p = T / rho, so pressure
    goes as the RECIPROCAL of the local radius of curvature, and on an ellipse that radius varies:

        rho(major-axis end) = b^2 / a      the SHARPEST curve — the sides of a wide, flat limb
        rho(minor-axis end) = a^2 / b      the flattest — front and back

    Their ratio is (a/b)^3, so a limb only slightly oval concentrates strap pressure sharply at its
    sides. This is why a cuff can pass a mean-pressure budget and still mark the skin in two lines.

    WHAT THIS DOES NOT COVER, and it is the important half: this is the law for a FLEXIBLE band in
    tension. A rigid shell that matches the section carries load in bearing instead, roughly evenly,
    and Laplace does not apply to it. So the factor here bounds the strap's share of the load, not
    the shell's — and the whole argument for contouring a cuff is that it moves load out of the term
    this function describes and into the term it does not.

    A circular shell on an oval limb is the worst case of both: it touches along two lines, carries
    no bearing load at all, and hands the entire job to the strap at this concentration.

    Returns the factor against the equal-perimeter CIRCLE, because that is the shape every mean
    pressure in this build was computed on.
    """
    circ = float(circ_mm)
    if circ <= 0:
        return {"ok": False, "error": "a circumference of zero describes nothing"}
    r_circle = circ / (2 * math.pi)
    width = float(width_mm or 0)
    if width <= 0:
        return {"ok": True, "measured": False, "aspect": 1.0, "peak_vs_circle": 1.0,
                "note": "no width measured — assumed round, which is the one shape a limb is not. "
                        "Measure the width across this site"}
    a = width / 2.0
    if a < r_circle:
        a = r_circle                                   # a near-round measurement; the solver snaps it
    # b from the perimeter, by the same Ramanujan bisection the geometry uses. Kept here rather than
    # imported so this module stays free of the CAD side; cross-checked in demo().
    lo, hi = 1e-6, a
    for _ in range(200):
        b = (lo + hi) / 2
        h = ((a - b) / (a + b)) ** 2
        per = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(max(0.0, 4 - 3 * h))))
        if per < circ:
            lo = b
        else:
            hi = b
    b = (lo + hi) / 2
    rho_min, rho_max = b ** 2 / a, a ** 2 / b
    return {"ok": True, "measured": True, "aspect": b / a,
            "semi_major_mm": a, "semi_minor_mm": b,
            "rho_min_mm": rho_min, "rho_max_mm": rho_max,
            "peak_vs_flat": rho_max / rho_min,          # (a/b)^3, side versus front
            "peak_vs_circle": r_circle / rho_min,       # what the mean-pressure figures assumed
            "peak_at": "the medial and lateral sides — the ends of the WIDE axis",
            "note": "applies to the strap's share of the load only; a shell that matches this "
                    "section carries its share in bearing, where Laplace does not apply"}


def prominences(region: str = "") -> dict:
    """Bony prominences that must be relieved, per region. Pressure here is not an average."""
    if region:
        v = BONY_PROMINENCES.get(region)
        if v is None:
            return {"ok": False, "error": f"unknown region '{region}'",
                    "known": sorted(BONY_PROMINENCES)}
        return {"ok": True, "region": region, "relieve": v}
    return {"ok": True, "regions": BONY_PROMINENCES}


def demo() -> None:
    """One runnable check. The first version of this test asserted 200 N through a 60 cm^2 cuff was
    25 mmHg. It is 250 mmHg -- a factor of ten, guessed instead of computed. That error is the whole
    reason this module exists: 20 mmHg is a far smaller budget than intuition says, and a cuff cannot
    be a load path.
    """
    r = interface_pressure(200, 60)
    assert r["ok"] and abs(r["pressure_mmHg"] - 250.0) < 0.5, r
    assert r["verdict"] == "FAIL" and "fix" in r, r
    # the area that would pass scales exactly with the pressure ratio
    assert abs(r["area_needed_cm2"]["prolonged"] - 60 * 250.0 / 20.0) < 1.0, r
    assert interface_pressure(20, 200)["verdict"] == "pass"
    assert interface_pressure(100, 0)["ok"] is False          # zero area is an error, not infinity
    assert interface_pressure(100, 50, site="nope")["ok"] is False

    # The site budget must actually change the verdict, or it is decoration: 150 N on 120 cm^2 is
    # 94 mmHg -- impossible on a cuff, acceptable on an ischial seat for a short bout.
    assert interface_pressure(150, 120, hours=2, site="cuff")["verdict"] == "FAIL"
    assert interface_pressure(150, 120, hours=2, site="seat")["verdict"] == "marginal"
    # a bony prominence can never pass, at any area
    assert interface_pressure(1, 10_000, site="bony")["verdict"] == "FAIL"

    # capacity is the inverse of pressure, and must agree with it
    cap = load_capacity(300, site="cuff")
    assert abs(cap["max_force_N"] - 80.0) < 0.5, cap
    back = interface_pressure(cap["max_force_N"], 300, site="cuff")
    assert abs(back["pressure_mmHg"] - 20.0) < 0.2, back
    assert load_capacity(120, site="bony")["max_force_N"] == 0.0
    assert load_capacity(0)["ok"] is False

    # ROM: target 75, slip 19.4, 5 deg detents -> ideal 55.6, must snap DOWN to 55, never up to 60.
    c = rom_calibrate(75)
    assert c["set_stop_to_deg"] == 55, c
    assert c["predicted_achieved_deg"] <= 75, c
    # rounding never grants extra motion, at any target
    for t in range(20, 121):
        k = rom_calibrate(t)
        if k["ok"]:
            assert k["predicted_achieved_deg"] <= t, (t, k)
    # a target inside the slip is impossible, and says so
    assert rom_calibrate(10)["ok"] is False
    assert rom_calibrate(75, step_deg=0)["ok"] is False

    # Dynamic: the seat that passes standing must be reported as marginal on stairs. If this ever
    # says "pass everywhere" for the v1 seat, the gait factors have been quietly removed.
    d = dynamic_check(200, 290, site="seat", hours=2, label="ischial seat")
    assert d["ok"] and d["worst_verdict"] != "pass", d
    by = {r["condition"]: r for r in d["conditions"]}
    assert by["standing"]["verdict"] == "pass", by["standing"]
    assert by["stairs"]["pressure_mmHg"] > by["standing"]["pressure_mmHg"] * 1.7, by
    # a pad big enough for the worst case must pass everywhere
    big = dynamic_check(200, d["area_for_worst_case_cm2"] + 1, site="seat", hours=2)
    assert big["passes_everywhere"], big
    assert dynamic_check(100, 0, site="cuff")["ok"] is False

    # ADL: the 75 deg hip cap must allow walking and stairs, and must block sitting and toileting.
    a = adl_check(75, 90)
    assert "walking, level" in a["allowed"] and "stair ascent" in a["allowed"], a["allowed"]
    for blocked in ("sitting, standard chair", "rising from a chair", "on/off a standard toilet"):
        assert blocked in a["blocked"], (blocked, a["blocked"])
    # an unrestricted hip blocks nothing on this list except what the KNEE limit blocks
    assert adl_check(180, 180)["blocked"] == [], adl_check(180, 180)["blocked"]
    assert "sitting in a bath" in adl_check(180, 90)["blocked"]

    # Offload gap: the frame carries load only by being long, and the sensitivity is the stiffness.
    g = offload_gap(200, tissue_k_n_per_mm=20.0, adjuster_pitch_mm=1.0, detent_pitch_mm=10.0)
    assert g["ok"] and abs(g["gap_mm"] - 10.0) < 0.2, g          # 200 N / ~20 N/mm
    # the whole point: a 10 mm detent moves the load by the entire design load, a thread does not
    assert g["detent_adequate"] is False, g
    assert g["adjuster_adequate"] is True, g
    assert abs(g["load_step_per_detent_N"] - 200) < 5, g
    assert g["load_step_per_quarter_turn_N"] < 10, g
    # a softer wearer needs a bigger gap for the same force, always
    prev = 0.0
    for kt in (10, 15, 20, 30, 40):
        r = offload_gap(200, tissue_k_n_per_mm=kt)
        assert prev == 0.0 or r["gap_mm"] < prev, (kt, r["gap_mm"], prev)
        prev = r["gap_mm"]
    # the stiff structure must not dominate the series stiffness
    assert abs(offload_gap(200, 20, upright_k_n_per_mm=1e9)["series_k_N_per_mm"] - 20) < 0.1
    assert offload_gap(0)["ok"] is False and offload_gap(200, 0)["ok"] is False

    # Seat height: a tighter hip limit must demand a HIGHER seat, always.
    s = seat_height(75, thigh_len_mm=430, popliteal_mm=430)
    assert s["ok"] and 130 < s["rise_above_knee_mm"] < 160, s     # ~15+5 deg of declination
    prev = -1.0
    for lim in range(30, 91, 5):
        r = seat_height(lim, 430, 430)
        assert r["rise_above_knee_mm"] < prev or prev < 0, (lim, r["rise_above_knee_mm"], prev)
        prev = r["rise_above_knee_mm"]
    assert seat_height(0)["ok"] is False and seat_height(120)["ok"] is False
    # curvature: the reason a contoured cuff is not a cosmetic preference
    rnd = curvature_factor(600.0, 0)
    assert rnd["ok"] and not rnd["measured"] and rnd["peak_vs_circle"] == 1.0
    oval = curvature_factor(600.0, 240.0)
    assert oval["aspect"] < 0.7, oval["aspect"]
    # sharpest curvature is at the ends of the WIDE axis, and the ratio is the aspect cubed
    assert abs(oval["peak_vs_flat"] - (1 / oval["aspect"]) ** 3) < 1e-6
    assert oval["peak_vs_circle"] > 1.9, oval["peak_vs_circle"]
    # a width at the circle's own diameter is round, and concentrates nothing
    flat = curvature_factor(600.0, 600.0 / math.pi)
    assert abs(flat["peak_vs_circle"] - 1.0) < 0.01, flat["peak_vs_circle"]
    print(f"curvature: a {600 / math.pi:.0f} mm-wide 600 mm limb is round; at 240 mm wide the "
          f"aspect is {oval['aspect']:.2f} and strap pressure at the sides is "
          f"{oval['peak_vs_circle']:.1f}x what the mean assumed")

    print("wearable.py: all checks pass")


if __name__ == "__main__":
    demo()
