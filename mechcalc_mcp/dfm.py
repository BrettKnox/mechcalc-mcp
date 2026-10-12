"""dfm.py — the things that are true about FDM and not about geometry.

A model that is dimensionally perfect prints as a part that does not fit. The gap is not error, it
is systematic: an extruder lays a bead of finite width along a path, and on the inside of a curve
that bead squeezes inward, so every hole comes out smaller than it was drawn. Draw a 4.2 mm bore
for an M4 pin and get 3.9 mm, and the pin does not go in.

Most CAD-for-printing pipelines know this and leave holes nominal anyway, which is the worst of both:
the numbers look authoritative and the parts do not fit.

TWO KINDS OF NUMBER LIVE HERE, and they are labelled differently on purpose:

  * MEASURED — `calibrate.hole_shrink` records what a coupon actually came out at, per material.
    Once that exists it is used and said so.
  * ASSUMED  — until then, a conservative default that is stated as a default every time it is
    applied. A compensation you cannot trace is worse than none, because it hides the error.

SCREWS GO INTO BRASS, NOT PLASTIC. Tapping a thread into PLA gives a few usable insertions before it
strips. A heat-set insert is a knurled brass sleeve pushed in with a soldering iron; it is the
standard answer and it needs a plain straight hole of a specific size, plus enough wall around it,
which is what `insert_boss` returns. Its hole IS compensated: an insert forms its own thread but not
its own pilot, and a hole modelled at the datasheet number prints undersize, so the insert has to
displace the difference into the seating face and its own thread.
"""
from __future__ import annotations

# Diametral shrink of a hole printed about the Z axis, in mm, until a coupon says otherwise.
# Conservative: real Ender-class printers land between 0.15 and 0.35 with a 0.4 nozzle.
ASSUMED_HOLE_SHRINK_MM = 0.20
# A hole printed on its side has no floor to sit on and sags at the top; it needs more.
ASSUMED_HORIZONTAL_EXTRA_MM = 0.15
# Brass needs material around it or the boss splits when the insert goes in.
INSERT_WALL_MM = 1.7
# A 45 deg lead-in at the mouth of an insert hole. It lets the insert start square and
# gives the first of the displaced plastic somewhere to go other than the seating face.
CHAMFER_MM = 0.5
# The first layer squashes outward. A chamfer this size on a bottom edge keeps the part flat-seating.
ELEPHANT_FOOT_CHAMFER_MM = 0.5


FASTENERS = {  # metric socket-head cap screw + its nut/washer/tap/clearance
    # proof_load_n and recommended_torque_nm are property class 8.8 (proof stress 580 MPa), with
    # the torque targeting 75% of proof at K=0.2 dry. Torque-only preload scatters about ±25%.
    "M2": {"clearance": 2.4, "tap": 1.6, "head_d": 3.8,  "head_h": 2.0, "hex": 1.5, "nut_af": 4.0,  "nut_thick": 1.6, "insert_hole": 3.2, "insert_depth": 3.5, "insert_len": 4.0, "tensile_area_mm2": 2.07, "proof_load_n": 1201, "recommended_torque_nm": 0.36},
    "M2.5": {"clearance": 2.9, "tap": 2.05, "head_d": 4.5, "head_h": 2.5, "hex": 2.0, "nut_af": 5.0,  "nut_thick": 2.0, "insert_hole": 3.5, "insert_depth": 4.0, "insert_len": 4.6, "tensile_area_mm2": 3.39, "proof_load_n": 1966, "recommended_torque_nm": 0.74},
    "M3": {"clearance": 3.4, "tap": 2.5, "head_d": 5.5,  "head_h": 3.0, "hex": 2.5, "nut_af": 5.5,  "nut_thick": 2.4, "insert_hole": 4.0, "insert_depth": 4.0, "insert_len": 5.7, "tensile_area_mm2": 5.03, "proof_load_n": 2917, "recommended_torque_nm": 1.31},
    "M4": {"clearance": 4.5, "tap": 3.3, "head_d": 7.0,  "head_h": 4.0, "hex": 3.0, "nut_af": 7.0,  "nut_thick": 3.2, "insert_hole": 5.6, "insert_depth": 5.5, "insert_len": 8.1, "tensile_area_mm2": 8.78, "proof_load_n": 5092, "recommended_torque_nm": 3.06},
    "M5": {"clearance": 5.5, "tap": 4.2, "head_d": 8.5,  "head_h": 5.0, "hex": 4.0, "nut_af": 8.0,  "nut_thick": 4.0, "insert_hole": 6.4, "insert_depth": 6.5, "insert_len": 9.5, "tensile_area_mm2": 14.2, "proof_load_n": 8236, "recommended_torque_nm": 6.18},
    "M6": {"clearance": 6.6, "tap": 5.0, "head_d": 10.0, "head_h": 6.0, "hex": 5.0, "nut_af": 10.0, "nut_thick": 5.0, "insert_hole": 8.0, "insert_depth": 8.0, "insert_len": 12.7, "tensile_area_mm2": 20.1, "proof_load_n": 11658, "recommended_torque_nm": 10.49},
    "M8": {"clearance": 9.0, "tap": 6.8, "head_d": 13.0, "head_h": 8.0, "hex": 6.0, "nut_af": 13.0, "nut_thick": 6.5, "insert_hole": 10.0, "insert_depth": 10.0, "insert_len": 12.7, "tensile_area_mm2": 36.6, "proof_load_n": 21228, "recommended_torque_nm": 25.47},
}


def _measured_for(material: str) -> float | None:
    """This machine's own hole shrink, if a printed part has ever been measured. Else None."""
    try:
        from . import calibrate
        rec = (calibrate.load() or {}).get(str(material or "pla").strip().lower()) or {}
        v = rec.get("hole_shrink_mm")
        return float(v) if v is not None else None
    except Exception:                                     # noqa: BLE001 — no calibration is normal
        return None


def hole(nominal_mm: float, *, material: str = "pla", axis: str = "z",
         measured_shrink_mm: float | None = None) -> dict:
    """The diameter to MODEL so the printed hole comes out at `nominal_mm`.

    Always returns the reason with the number, because a silently adjusted dimension is
    indistinguishable from a mistake when somebody reads the model six months later.
    """
    n = float(nominal_mm)
    if n <= 0:
        return {"ok": False, "error": "a hole needs a positive diameter"}
    if measured_shrink_mm is None:
        # A number this machine actually produced beats a textbook one, so look before assuming.
        # This is the whole point of printing a part with holes in it: the next part fits better.
        measured_shrink_mm = _measured_for(material)
    if measured_shrink_mm is None:
        shrink, source = ASSUMED_HOLE_SHRINK_MM, "assumed"
    else:
        shrink, source = float(measured_shrink_mm), "measured"
    extra = ASSUMED_HORIZONTAL_EXTRA_MM if str(axis).lower() != "z" else 0.0
    model = round(n + shrink + extra, 2)
    why = (f"a {n:g} mm hole prints about {shrink:g} mm undersize"
           + (f", plus {extra:g} mm because it is printed on its side" if extra else "")
           + f" ({source})")
    return {"ok": True, "nominal_mm": n, "model_mm": model, "shrink_mm": shrink,
            "source": source, "axis": axis, "why": why,
            "note": ("this is a default, not your printer — print the bore coupon and record it "
                     "with calibrate.hole_shrink" if source == "assumed" else "")}


def shaft(nominal_mm: float, *, material: str = "pla", clearance_mm: float = 0.0,
          measured_growth_mm: float | None = None) -> dict:
    """The diameter to MODEL so a printed peg, pin or spigot comes out at `nominal_mm`.

    The mirror of `hole`, and the half that catches people out. Same cause, opposite sign: the bead
    has width, so on the OUTSIDE of a curve it spreads outward and the feature grows. A pin drawn at
    4.0 mm prints at about 4.2 and will not enter the 4.3 mm bore it was designed for — the hole
    shrank and the pin grew, and the two errors add.

    `clearance_mm` is running clearance taken off on top, for a pin something has to turn on.
    """
    n = float(nominal_mm)
    if n <= 0:
        return {"ok": False, "error": "a shaft needs a positive diameter"}
    if measured_growth_mm is None:
        measured_growth_mm = _measured_for(material)      # the same bead width, so the same number
    if measured_growth_mm is None:
        growth, source = ASSUMED_HOLE_SHRINK_MM, "assumed"
    else:
        growth, source = float(measured_growth_mm), "measured"
    model = round(n - growth - float(clearance_mm), 2)
    if model <= 0:
        return {"ok": False, "error": f"{n:g} mm is too small to compensate by {growth:g} mm"}
    return {"ok": True, "nominal_mm": n, "model_mm": model, "growth_mm": growth,
            "clearance_mm": float(clearance_mm), "source": source,
            "why": (f"a {n:g} mm peg prints about {growth:g} mm oversize ({source})"
                    + (f", less {clearance_mm:g} mm so it turns freely" if clearance_mm > 0 else "")
                    + (f", plus {-clearance_mm:g} mm of interference so it grips" if clearance_mm < 0
                       else "")),
            "note": ("this is a default, not your printer — the same coupon that gives hole shrink "
                     "gives this" if source == "assumed" else "")}


def insert_boss(screw: str = "M3", *, material: str = "pla",
                measured_shrink_mm: float | None = None) -> dict:
    """The hole and the boss for a heat-set brass insert, so a real screw has something to bite.

    THIS HOLE IS COMPENSATED, and it used to not be. The old reasoning was that an insert melts its
    own fit so the hole can stay nominal — which confuses two different things. The insert forms its
    own THREAD, yes; it does not choose its own PILOT. A printed hole still comes out undersize, so
    modelling the datasheet's 4.0 mm gives a real hole nearer 3.8, and the insert then has to
    displace the difference as it goes in.

    Molten PLA is not compressible, so that extra volume has to leave: it flashes up onto the seating
    face as a raised collar, which stops the joint sitting flat, and it runs into the insert's own
    thread, which seizes the screw. Getting it in at all needs more force and a longer dwell, and PLA
    degrades at the 210-230 C the iron runs at. The failure is not "the insert is loose" -- it is a
    part that no longer seats and a thread full of plastic.

    Oversizing is the opposite error and still real: too big and there is nothing for the knurls to
    bite. The answer is to aim at the datasheet number and hit it, which is what compensation is for.
    """
    f = FASTENERS.get(screw)
    if not f or not f.get("insert_hole"):
        return {"ok": False, "error": f"no heat-set insert data for {screw!r}",
                "have": sorted(k for k, v in FASTENERS.items() if v.get("insert_hole"))}
    want = float(f["insert_hole"])
    comp = hole(want, material=material, axis="z", measured_shrink_mm=measured_shrink_mm)
    d = comp["model_mm"]
    # DEPTH FOLLOWS THE INSERT'S LENGTH, not a number of its own. The table used to carry a bare
    # `insert_depth` with no stated origin, and for M3 it was 4.0 mm against a part that is commonly
    # 5.7 mm long — an insert standing 1.2 mm proud of a face that is supposed to seat flat.
    ln = float(f.get("insert_len") or f.get("insert_depth") or 0.0)
    depth = round(ln + 0.6, 2)          # somewhere for the displaced plastic to go
    return {"ok": True, "screw": screw, "hole_mm": d, "printed_mm": want,
            "depth_mm": depth, "insert_len_mm": ln,
            "boss_od_mm": round(d + 2 * INSERT_WALL_MM, 2),
            # A lead-in gives the insert somewhere to start square, and the counterbore at the
            # bottom takes the plastic the knurls displace instead of pushing it up the thread.
            "chamfer_mm": round(CHAMFER_MM, 2), "relief_dia_mm": round(want + 0.8, 2),
            "why": (f"a {screw} heat-set insert is about {ln:g} mm long and wants a {want:g} mm "
                    f"pilot, so model {d:g} mm to print at {want:g} ({comp['source']} shrink), "
                    f"{depth:g} mm deep, with {INSERT_WALL_MM:g} mm of wall so the boss does not split"),
            "how": (f"chamfer the mouth {CHAMFER_MM:g} mm at 45 deg, then push the insert in with a "
                    f"soldering iron at about 210 C, square to the face, until flush"),
            "compensated": (f"modelled {d:g} to land on the datasheet's {want:g}. An UNcompensated "
                            f"{want:g} prints near {round(want - comp['shrink_mm'], 2):g}, and the "
                            f"insert has to displace the difference -- which flashes onto the "
                            f"seating face and runs into its own thread"),
            "check_the_packet": ("lengths differ between suppliers — measure one and correct "
                                 "insert_len in parts.FASTENERS if it is not "
                                 f"{ln:g} mm")}


def wall(target_mm: float, nozzle_mm: float = 0.4) -> dict:
    """The nearest wall thickness that is a whole number of extrusions.

    A 1.5 mm wall on a 0.4 nozzle is three 0.42 mm beads and a 0.24 mm gap the slicer fills with
    nothing, or widens the beads to cover. Either way it is weaker than the 1.26 mm it should have
    been, and thicker than the 1.68 mm it could have been.
    """
    ew = round(nozzle_mm * 1.05, 3)                       # extrusion width, the usual default
    n = max(1, round(float(target_mm) / ew))
    got = round(n * ew, 2)
    return {"ok": True, "asked_mm": float(target_mm), "wall_mm": got, "beads": n,
            "extrusion_width_mm": ew,
            "why": f"{n} x {ew:g} mm beads; the slicer cannot lay part of one"}


def review(part: dict, shop: dict | None = None) -> dict:
    """The standing checklist, applied to one part. Advice, never a refusal.

    Everything here is a rule of thumb that a person can overrule with a reason, so it reports and
    does not block. The ones that would actually stop a print live in `buildability`.
    """
    sh = shop or {}
    nozzle = float(sh.get("nozzle_mm") or 0.4)
    max_over = float(sh.get("max_overhang_deg") or 50.0)
    bbox = part.get("bbox_mm") or []
    notes = []

    if bbox and len(bbox) >= 3:
        h, foot = float(bbox[2]), max(float(bbox[0]), float(bbox[1]))
        if h > 0 and foot > 0 and h / foot > 3.0:
            notes.append({"rule": "tall and thin", "severity": "warn",
                          "what": f"{h:g} mm tall on a {foot:g} mm footprint",
                          "fix": "add a brim, or lie it down — it will be knocked over by the nozzle"})
        notes.append({"rule": "layer direction", "severity": "info",
                      "what": "a printed part is weakest across the layers, by roughly half",
                      "fix": "orient it so the load runs along the layers, not across them"})

    notes.append({"rule": "bottom edges", "severity": "info",
                  "what": f"the first layer spreads; a {ELEPHANT_FOOT_CHAMFER_MM:g} mm chamfer on "
                          f"bottom edges keeps it seating flat",
                  "fix": "chamfer, or turn on elephant-foot compensation in the slicer"})
    notes.append({"rule": "overhangs", "severity": "info",
                  "what": f"anything past {max_over:g} degrees from vertical needs support",
                  "fix": "chamfer it to 45 degrees instead, and print it without support"})
    notes.append({"rule": "holes for screws", "severity": "info",
                  "what": "a thread tapped into PLA strips after a few insertions",
                  "fix": "use a heat-set brass insert — dfm.insert_boss('M3') gives the hole"})
    if float(sh.get("min_wall_mm") or 0.8) < nozzle * 2:
        notes.append({"rule": "minimum feature", "severity": "info",
                      "what": f"nothing thinner than {nozzle:g} mm can be printed at all, and under "
                              f"{nozzle * 2:g} mm it is a single bead with no strength",
                      "fix": f"keep structural walls at or above {nozzle * 2:g} mm"})
    return {"ok": True, "part": part.get("name") or part.get("key") or "", "notes": notes}


def _selfcheck() -> None:
    h = hole(4.2)
    assert h["model_mm"] == 4.4, h
    assert h["source"] == "assumed" and "coupon" in h["note"]
    # a hole on its side needs more, and says why
    side = hole(4.2, axis="x")
    assert side["model_mm"] == 4.55 and "on its side" in side["why"]
    # a measured number wins and is labelled as such
    m = hole(4.2, measured_shrink_mm=0.31)
    assert m["model_mm"] == 4.51 and m["source"] == "measured" and not m["note"]
    assert not hole(0)["ok"]

    b = insert_boss("M3")
    assert b["printed_mm"] == 4.0, "the datasheet pilot is what has to come out of the printer"
    assert b["hole_mm"] > b["printed_mm"], "so the MODELLED hole has to be bigger"
    assert b["depth_mm"] > b["insert_len_mm"] and b["chamfer_mm"] > 0
    assert not insert_boss("M99")["ok"]

    w = wall(1.5)
    assert w["beads"] == 4 and w["wall_mm"] == 1.68, w
    assert wall(0.4)["beads"] == 1

    r = review({"name": "post", "bbox_mm": [10, 10, 60]}, {"nozzle_mm": 0.4})
    rules = {n["rule"] for n in r["notes"]}
    assert "tall and thin" in rules and "holes for screws" in rules
    print("dfm ok — a 4.2 mm bore is modelled at 4.4 and says the number is assumed, an M3 insert "
          "hole is modelled 4.2 to PRINT at the datasheet's 4.0, and a 1.5 mm wall is really 1.68")


if __name__ == "__main__":
    _selfcheck()
