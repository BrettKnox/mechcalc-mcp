"""calibrate.py — turn a printed test beam into YOUR printer's real numbers.

WHY THIS EXISTS. Every stiffness and stress number this app produces rests on `matprops`, whose
polymer rows are typical datasheet values. A datasheet describes injection-moulded resin. Your part
is a stack of partially fused extrusions made on a specific machine, at a specific layer height, in
a specific orientation, out of a spool that may have absorbed water. The real modulus is commonly
20-40% off the datasheet, and across layers it can be half. Nothing downstream can be trusted more
than the material data underneath it, so this is the cheapest way to make everything else honest.

WHAT IT MEASURES. Three-point bend, not cantilever. A cantilever needs a perfectly rigid clamp, and
at a kitchen table the clamp compliance quietly ends up in your answer as a softer material. Three
points need no clamp: two supports, load in the middle.

    delta = P L^3 / (48 E I),      I = b h^3 / 12,      sigma_max = 3 P L / (2 b h^2)

THE SLOPE TRICK, which is the whole reason this gives usable numbers with hardware you own. Do NOT
load once and divide. Load in four steps and fit a line to (load, deflection); use only the SLOPE.
The intercept absorbs seating error, support settling and the fact that your zero was never truly
zero, and those are the dominant errors in a home rig. The intercept is reported so you can see how
much error it soaked up, and a large one is flagged rather than hidden.

WHAT IT CANNOT TELL YOU. Bending gives an effective flexural modulus, not a true tensile modulus,
and it is dominated by the outer fibres, which on a printed part are the perimeters. That is
arguably the number you want for a printed part, but it is not the same quantity a datasheet lists,
which is why the derived `source` string says so instead of pretending.

Run it: python -m mechcalc.calibrate
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("MECHCALC_DATA_DIR", Path.home() / ".mechcalc"))

STORE = DATA_DIR / "calibration.json"

# The coupon. Chosen so a full water bottle bends it far enough to measure with calipers while the
# peak stress stays well inside elastic for every printable polymer in the table.
COUPON = {
    "length_mm": 120.0,      # 10 mm overhang each side of the span
    "span_mm": 100.0,
    "width_mm": 12.0,
    "thickness_mm": 3.0,
    "count": 3,              # per orientation; three is enough to see scatter, five is better
    "infill_pct": 100,
    "perimeters": 4,
    "layer_mm": 0.2,
}

# Print the SAME beam three ways. The difference between them is the anisotropy you cannot get from
# any datasheet, and it is what decides whether a bracket should be printed lying down or standing up.
ORIENTATIONS = {
    "flat":    "lying on the bed, 12 mm face down. Layers run along the beam; the strongest way.",
    "on_edge": "lying on the bed, 3 mm face down. Layers still run along the beam but the wall stack "
               "carries the bending stress.",
    "upright": "standing on the 12x3 end, printed as a tower. Layers run ACROSS the beam so bending "
               "pulls the layer bonds apart. Expect this one to be much weaker, and print a brim.",
}

# Loads to apply, in kilograms. A 500 ml water bottle is 0.5 kg to within a couple of grams, which is
# better accuracy than most kitchen scales and costs nothing.
LOAD_STEPS_KG = (0.5, 1.0, 1.5, 2.0)
G = 9.80665

# Above this the fitted line is not a line: something slipped, yielded, or the supports moved.
MIN_R2 = 0.99
# r^2 is a weak curvature detector: four points on a gentle curve still fit a line at r^2 > 0.99.
# Comparing the first pair's slope with the last pair's catches the yield knee that actually happens.
MAX_SLOPE_DRIFT = 0.30
# An intercept this large relative to the last deflection means the rig, not the material, dominates.
MAX_INTERCEPT_FRAC = 0.25


def second_moment_mm4(width_mm: float, thickness_mm: float) -> float:
    """I for a rectangle bent about its neutral axis. Thickness is CUBED, so measure it with
    calipers rather than trusting the slicer: 3.0 vs 3.2 mm is a 20% error in I."""
    return width_mm * thickness_mm ** 3 / 12.0


def deflection_mm(load_n: float, span_mm: float, e_mpa: float, i_mm4: float) -> float:
    """Predicted centre deflection of a simply supported beam, central point load."""
    return load_n * span_mm ** 3 / (48.0 * e_mpa * i_mm4)


def stress_mpa(load_n: float, span_mm: float, width_mm: float, thickness_mm: float) -> float:
    """Peak fibre stress at mid-span, central point load."""
    return 3.0 * load_n * span_mm / (2.0 * width_mm * thickness_mm ** 2)


def modulus_from_slope(slope_mm_per_n: float, span_mm: float, i_mm4: float) -> float:
    """E from the load-deflection SLOPE. Inverting delta = P L^3 / (48 E I)."""
    if slope_mm_per_n <= 0:
        raise ValueError("slope must be positive: deflection has to grow with load")
    return span_mm ** 3 / (48.0 * i_mm4 * slope_mm_per_n)


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares slope, intercept and r^2. Small enough to not warrant scipy, and this has to
    run at the `numpy` capability tier."""
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two load steps; one point cannot separate seating error")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("every load step is the same value")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return slope, intercept, r2


@dataclass
class Run:
    """One coupon, loaded through several steps."""
    material: str
    orientation: str
    width_mm: float
    thickness_mm: float
    span_mm: float
    loads_kg: list[float]
    deflections_mm: list[float]
    failed_at_kg: float | None = None      # if you kept going until it broke
    notes: str = ""
    warnings: list[str] = field(default_factory=list)

    def result(self) -> dict:
        if len(self.loads_kg) != len(self.deflections_mm):
            raise ValueError("each load step needs exactly one deflection reading")
        i = second_moment_mm4(self.width_mm, self.thickness_mm)
        xs = [kg * G for kg in self.loads_kg]
        slope, intercept, r2 = _fit(xs, self.deflections_mm)
        e = modulus_from_slope(slope, self.span_mm, i)

        warn = []
        if len(xs) >= 4:
            s_lo = (self.deflections_mm[1] - self.deflections_mm[0]) / (xs[1] - xs[0])
            s_hi = (self.deflections_mm[-1] - self.deflections_mm[-2]) / (xs[-1] - xs[-2])
            drift = abs(s_hi - s_lo) / abs(s_lo) if s_lo else 0.0
            if drift > MAX_SLOPE_DRIFT:
                warn.append(f"the response is not linear: the last step is {drift*100:.0f}% "
                            f"{'softer' if s_hi > s_lo else 'stiffer'} than the first. If it softened "
                            f"the beam is yielding, so drop the top load. If it stiffened, the beam "
                            f"was not seated. Either way E from this run is wrong.")
        if r2 < MIN_R2:
            warn.append(f"r^2 {r2:.3f} below {MIN_R2}: the response is not linear. Something slipped, "
                        f"yielded, or a support moved. Re-run before trusting E.")
        # compare the intercept against the deflection RANGE, not the largest reading: the largest
        # reading contains the intercept, so it dilutes exactly the error being tested for. The range
        # is the part of the signal the material actually produced.
        signal = abs(slope) * (max(xs) - min(xs)) or 1e-9
        if abs(intercept) > MAX_INTERCEPT_FRAC * signal:
            warn.append(f"intercept {intercept:.2f} mm is {abs(intercept)/signal*100:.0f}% of the "
                        f"{signal:.2f} mm the load actually produced: the rig dominates. Seat the "
                        f"beam under a small preload and re-zero.")
        if max(self.deflections_mm) < 1.0:
            warn.append("largest deflection under 1 mm: calipers cannot resolve this well. Use a "
                        "longer span or a thinner beam.")
        out = {
            "material": self.material, "orientation": self.orientation,
            "E_mpa": round(e, 1),
            "slope_mm_per_n": round(slope, 5),
            "intercept_mm": round(intercept, 3),
            "r2": round(r2, 4),
            "i_mm4": round(i, 3),
            "n_points": len(xs),
            "max_stress_mpa": round(stress_mpa(max(xs), self.span_mm, self.width_mm,
                                               self.thickness_mm), 2),
            "warnings": warn + list(self.warnings),
            "notes": self.notes,
        }
        if self.failed_at_kg:
            out["flexural_strength_mpa"] = round(
                stress_mpa(self.failed_at_kg * G, self.span_mm, self.width_mm, self.thickness_mm), 2)
        return out


def plan(material: str, bed_mm: tuple[float, float] | None = None) -> dict:
    """What to print and exactly what to do with it. Data, so a UI or a report can render
    it and neither has to restate the protocol."""
    from . import matprops
    m = matprops.get(material)                     # raises on an unknown material, deliberately
    i = second_moment_mm4(COUPON["width_mm"], COUPON["thickness_mm"])
    peak_n = max(LOAD_STEPS_KG) * G
    predicted = deflection_mm(peak_n, COUPON["span_mm"], m.E_mpa, i)
    peak_sigma = stress_mpa(peak_n, COUPON["span_mm"], COUPON["width_mm"], COUPON["thickness_mm"])

    issues = []
    if predicted < 1.0:
        issues.append(f"at {m.name}'s datasheet stiffness the beam only moves {predicted:.2f} mm at "
                      f"{max(LOAD_STEPS_KG)} kg, which calipers cannot resolve. Add load or thin the beam.")
    if peak_sigma > m.allowable_mpa:
        issues.append(f"peak stress {peak_sigma:.1f} MPa exceeds {m.name}'s allowable "
                      f"{m.allowable_mpa} MPa: it may yield mid-test and the fit will curve. "
                      f"Drop the top load step.")
    if bed_mm and (COUPON["length_mm"] > max(bed_mm) or COUPON["width_mm"] > min(bed_mm)):
        issues.append(f"the {COUPON['length_mm']:.0f} mm beam does not fit a {bed_mm[0]:.0f}x"
                      f"{bed_mm[1]:.0f} mm bed. Print it diagonally or shorten the span.")

    return {
        "material": m.key, "material_name": m.name,
        "coupon": dict(COUPON),
        "orientations": dict(ORIENTATIONS),
        "load_steps_kg": list(LOAD_STEPS_KG),
        "predicted_deflection_mm": round(predicted, 2),
        "peak_stress_mpa": round(peak_sigma, 1),
        "datasheet_E_mpa": m.E_mpa,
        "total_coupons": COUPON["count"] * len(ORIENTATIONS),
        "issues": issues,
        "steps": [
            f"Print {COUPON['count']} beams in EACH of the three orientations "
            f"({COUPON['count'] * len(ORIENTATIONS)} total), {COUPON['layer_mm']} mm layers, "
            f"{COUPON['perimeters']} perimeters, {COUPON['infill_pct']}% infill, same spool, same day.",
            "Measure each beam's real width and thickness with calipers at three points and average. "
            "Do not use the nominal numbers: thickness is cubed, so a 0.2 mm error is a 20% error in I.",
            f"Set two supports exactly {COUPON['span_mm']:.0f} mm apart. Two bolts, two pencils, the "
            f"edges of two books. They only need to be parallel and not move.",
            "Rest the beam on the supports and hang a small bag from the mid-point. Zero your "
            "measurement with the empty bag on, so the bag's own weight is not in the answer.",
            f"Add load in steps of {', '.join(str(x) for x in LOAD_STEPS_KG)} kg (a 500 ml water "
            f"bottle is 0.5 kg). At each step, measure how far the mid-point dropped.",
            "Enter all four points. The fit uses the SLOPE, so a small zeroing error cancels out. "
            "One point would not let it.",
            "Optional but worth it: on one beam per orientation, keep adding load until it breaks and "
            "record the weight. That gives real flexural strength, which sets the knockdown factor.",
        ],
    }


# ── persistence: a JSON overlay next to the database, not a migration ────────────────────────────
def load() -> dict:
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(data: dict) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def record(runs: list[Run], printer: str = "", stamp: str = "") -> dict:
    """Fold a batch of coupon runs into the stored calibration for their material.

    Orientations are kept separate rather than averaged, because averaging the strong and weak
    directions produces a number that describes no real part.
    """
    if not runs:
        raise ValueError("no runs")
    mats = {r.material for r in runs}
    if len(mats) != 1:
        raise ValueError(f"one material per batch, got {sorted(mats)}")
    material = runs[0].material

    by_orient: dict[str, list[dict]] = {}
    for r in runs:
        by_orient.setdefault(r.orientation, []).append(r.result())

    summary = {}
    for orient, results in by_orient.items():
        es = [x["E_mpa"] for x in results]
        mean = sum(es) / len(es)
        sd = math.sqrt(sum((e - mean) ** 2 for e in es) / len(es)) if len(es) > 1 else 0.0
        strengths = [x["flexural_strength_mpa"] for x in results if "flexural_strength_mpa" in x]
        entry = {"E_mpa": round(mean, 1), "sd_mpa": round(sd, 1),
                 "cv_pct": round(100 * sd / mean, 1) if mean else 0.0,
                 "n": len(es), "runs": results}
        if strengths:
            entry["flexural_strength_mpa"] = round(sum(strengths) / len(strengths), 2)
        if entry["cv_pct"] > 15:
            entry["warning"] = (f"{entry['cv_pct']:.0f}% scatter across {len(es)} coupons. That is "
                                f"print quality, not material. Check for warping or under-extrusion "
                                f"before using this number.")
        summary[orient] = entry

    data = load()
    data[material] = {"printer": printer, "measured_at": stamp, "orientations": summary}
    save(data)
    return data[material]


def record_hole(material: str, nominal_mm: float, measured_mm: float, printer: str = "",
                stamp: str = "") -> dict:
    """One measured bore on a printed coupon -> this material's real hole shrink.

    Until this existed there was nowhere to put the number. The print plan puts a shrink coupon first
    in every build and the bench task says "measure the 8 mm bore", and the answer had no home, so
    `tolerance.fit` went on assuming 0.25 mm forever and every printed fit reported "go measure your
    printer" to someone who already had.

    Shrink is the amount the hole came out UNDER nominal. A bore that measured larger than nominal is
    a real result too (over-compensated slicer settings) and is stored as a negative.
    """
    mat = str(material or "").strip().lower()
    if not mat:
        raise ValueError("a material is required")
    nominal, got = float(nominal_mm), float(measured_mm)
    if nominal <= 0 or got <= 0:
        raise ValueError("nominal and measured must both be positive millimetres")
    shrink = round(nominal - got, 4)
    if abs(shrink) > nominal * 0.5:
        raise ValueError(f"{shrink:+.2f} mm on a {nominal:g} mm bore is not a shrink, it is a "
                         f"different hole — check the numbers are the right way round")
    data = load()
    rec = data.get(mat) or {}
    when = stamp or datetime.now(timezone.utc).isoformat(timespec="seconds")

    # EVERY point is kept, not just the last one. A single bore gives one number with no way to know
    # whether it is representative; a set of them across different diameters shows whether the shrink
    # is a constant or grows with size, and that difference decides whether one compensation can be
    # applied to every hole. Re-measuring the same nominal replaces that point rather than adding a
    # second opinion.
    pts = [p for p in (rec.get("hole_points") or []) if abs(float(p["nominal_mm"]) - nominal) > 1e-6]
    pts.append({"nominal_mm": nominal, "measured_mm": got, "shrink_mm": shrink,
                "printer": printer or rec.get("printer", ""), "at": when})
    pts.sort(key=lambda p: p["nominal_mm"])

    shrinks = [float(p["shrink_mm"]) for p in pts]
    mean = round(sum(shrinks) / len(shrinks), 4)
    spread = round(max(shrinks) - min(shrinks), 4)
    rec["hole_points"] = pts
    rec["hole_shrink_mm"] = mean
    rec["hole_shrink_spread_mm"] = spread
    rec["hole_shrink_n"] = len(pts)
    # A wide spread means one number is the wrong model, so say it instead of averaging it away.
    rec["hole_shrink_note"] = (
        f"from {len(pts)} bore(s), {min(p['nominal_mm'] for p in pts):g}-"
        f"{max(p['nominal_mm'] for p in pts):g} mm"
        + (f"; they disagree by {spread:g} mm, which is more than the shrink itself — the shrink is "
           f"not one constant across those sizes" if len(pts) > 1 and spread > abs(mean) else ""))
    rec["hole_measured_at"] = when
    rec["hole_coupon"] = {"nominal_mm": nominal, "measured_mm": got, "printer": printer or
                          rec.get("printer", "")}
    data[mat] = rec
    save(data)
    return rec


def record_print_rate(material: str, estimated_min: float, actual_min: float,
                      what: str = "", printer: str = "", stamp: str = "") -> dict:
    """What a slicer said against what the plan predicted, so the next estimate is less wrong.

    The plan's minutes are material volume divided by a flow rate, which is a fair model for a solid
    brick and a poor one for anything else. A set of gears is almost entirely perimeter, eight
    separate objects means travelling between them on every layer, and a part small enough to need
    cooling time is paced by that rather than by flow. Measured on the first real print: 223 minutes
    predicted, 361 actual.

    Stored as a RATIO rather than a new rate constant, because the volume model is not wrong about
    volume — it is missing everything that is not volume, and that scales with how fiddly the set is.
    """
    est, act = float(estimated_min), float(actual_min)
    mat = str(material or "").strip().lower()
    if not mat:
        raise ValueError("a material is required")
    if est <= 0 or act <= 0:
        raise ValueError("both times must be positive minutes")
    data = load()
    rec = data.get(mat) or {}
    pts = list(rec.get("print_rate_points") or [])
    pts.append({"estimated_min": round(est, 1), "actual_min": round(act, 1),
                "ratio": round(act / est, 3), "what": str(what or "")[:120],
                "printer": printer or rec.get("printer", ""),
                "at": stamp or datetime.now(timezone.utc).isoformat(timespec="seconds")})
    ratios = [p["ratio"] for p in pts]
    rec["print_rate_points"] = pts[-12:]
    rec["print_time_factor"] = round(sum(ratios) / len(ratios), 3)
    rec["print_time_n"] = len(pts)
    data[mat] = rec
    save(data)
    return rec


def print_time_factor(material: str = "pla") -> float:
    """Multiplier for a volume-model print estimate. 1.0 when nothing has been measured."""
    try:
        rec = (load() or {}).get(str(material or "pla").strip().lower()) or {}
        return float(rec.get("print_time_factor") or 1.0)
    except Exception:                                     # noqa: BLE001
        return 1.0


def overlay() -> dict:
    """{material_key: {E_mpa, knockdown, z_strength_ratio, source}} for whatever has been measured.

    `flat` is the reference orientation because that is how most brackets get printed. The z ratio is
    the measured upright/flat ratio, which is the honest version of a number the table currently
    guesses at.
    """
    out = {}
    for mat, rec in load().items():
        orients = rec.get("orientations") or {}
        flat = orients.get("flat") or next(iter(orients.values()), None)
        if not flat or not flat.get("E_mpa"):
            continue
        patch = {"E_mpa": flat["E_mpa"]}
        up = orients.get("upright") or {}
        if up.get("E_mpa"):
            patch["z_strength_ratio"] = round(up["E_mpa"] / flat["E_mpa"], 3)
        where = rec.get("printer") or "this printer"
        when = rec.get("measured_at") or "undated"
        patch["source"] = (f"MEASURED on {where} ({when}): 3-point bend, n={flat.get('n', 0)}, "
                           f"CV {flat.get('cv_pct', 0)}%. Flexural modulus, not tensile.")
        out[mat] = patch
    return out


def _selfcheck() -> None:
    # hand calculation: b=12, h=3 -> I = 12*3^3/12 = 27 mm^4 exactly
    i = second_moment_mm4(12.0, 3.0)
    assert abs(i - 27.0) < 1e-9, i
    # E=3500, L=100, P=19.6133 N -> delta = P L^3 / (48 E I) = 19.6133e6 / 4_536_000
    p = 2.0 * G
    d = deflection_mm(p, 100.0, 3500.0, i)
    assert abs(d - 4.3239) < 1e-3, d
    # sigma = 3 P L / (2 b h^2) = 3*19.6133*100 / (2*12*9) = 27.24 MPa
    assert abs(stress_mpa(p, 100.0, 12.0, 3.0) - 27.24) < 0.01

    # a perfect run must recover the modulus it was generated from
    loads = list(LOAD_STEPS_KG)
    perfect = [deflection_mm(kg * G, 100.0, 3500.0, i) for kg in loads]
    r = Run("pla", "flat", 12.0, 3.0, 100.0, loads, perfect).result()
    assert abs(r["E_mpa"] - 3500.0) < 1.0, r["E_mpa"]
    assert r["r2"] > 0.999 and not r["warnings"], r

    # THE POINT OF THE SLOPE FIT: a constant zeroing error must not change the answer
    offset = [x + 0.9 for x in perfect]
    r2 = Run("pla", "flat", 12.0, 3.0, 100.0, loads, offset).result()
    assert abs(r2["E_mpa"] - 3500.0) < 1.0, "a seating offset leaked into E"
    assert any("intercept" in w for w in r2["warnings"]), "a 0.9 mm intercept must be reported"

    # a yield knee (the real failure mode) must be refused, not averaged into a plausible number
    knee = perfect[:3] + [perfect[3] * 1.8]
    kw = Run("pla", "flat", 12.0, 3.0, 100.0, loads, knee).result()["warnings"]
    assert any("not linear" in w and "softer" in w for w in kw), kw
    # and so must a gentle curve that r^2 alone would wave through
    bent = [x * (1 + 0.35 * k) for k, x in enumerate(perfect)]
    br = Run("pla", "flat", 12.0, 3.0, 100.0, loads, bent).result()
    assert any("not linear" in w for w in br["warnings"]), (br["r2"], br["warnings"])

    # breaking one gives flexural strength
    broke = Run("pla", "flat", 12.0, 3.0, 100.0, loads, perfect, failed_at_kg=4.0).result()
    assert abs(broke["flexural_strength_mpa"] - 54.48) < 0.05, broke["flexural_strength_mpa"]

    # the plan is honest about a material the coupon cannot measure well
    pl = plan("pla")
    assert pl["total_coupons"] == 9 and pl["datasheet_E_mpa"] == 3500
    assert abs(pl["predicted_deflection_mm"] - 4.32) < 0.02
    try:
        plan("unobtainium")
        raise AssertionError("an unknown material must raise, not substitute one")
    except Exception as e:
        assert "unknown material" in str(e)

    # overlay shape, without touching the real store
    global load
    _real = load
    load = lambda: {"pla": {"printer": "P1S", "measured_at": "2026-07-27", "orientations": {  # noqa: E731
        "flat": {"E_mpa": 2600.0, "n": 3, "cv_pct": 4.1},
        "upright": {"E_mpa": 1300.0, "n": 3, "cv_pct": 9.0}}}}
    ov = overlay()
    load = _real
    assert ov["pla"]["E_mpa"] == 2600.0 and ov["pla"]["z_strength_ratio"] == 0.5
    assert ov["pla"]["source"].startswith("MEASURED on P1S")

    print(f"calibrate ok - slope fit recovers E to <1 MPa and rejects a 0.9 mm seating offset "
          f"({COUPON['span_mm']:.0f} mm span, {max(LOAD_STEPS_KG)} kg -> {pl['predicted_deflection_mm']} mm on PLA)")


if __name__ == "__main__":
    _selfcheck()
