"""planetary.py — the arithmetic a planetary gearset either satisfies or does not.

Asked for a test that exercises the whole system rather than a flat plate: several parts that must
fit each other, real precision, few enough to print in an evening. A planetary gearset is the right
shape of problem, because almost every way of getting it wrong is silent until it is on the bench.

FOUR THINGS HAVE TO BE TRUE AT ONCE, and only the first is obvious:

  1. teeth fit          ring = sun + 2 x planet     or the planets do not span the gap
  2. planets share      (sun + ring) % n == 0       or they cannot sit at equal angles
  3. planets clear      adjacent tip circles must not overlap
  4. no undercut        below ~17 teeth at 20 degrees the cutter eats the flank root

The first pick tried here was sun 16, planet 16, ring 48 with three planets. It passes (1) — 48 is
16 + 32 — and fails (2), because 64 is not divisible by 3. The gears mesh perfectly and the third
planet has nowhere to go. Nothing in the geometry says so; the arithmetic does.

Everything downstream is derived from the same numbers, so the printed parts cannot disagree with
each other: centre distance sets the carrier's pin circle, tooth counts set the ring's bore.
"""
from __future__ import annotations

import math

# Below this, a standard 20 degree cutter undercuts the tooth root and the flank loses its involute.
MIN_TEETH = 17
# How much air between the tip circles of two neighbouring planets, in modules.
PLANET_GAP_M = 0.5


def design(module: float, sun: int, planet: int, n_planets: int = 3, *,
           pressure_angle: float = 20.0, face_mm: float = 8.0, backlash_mm: float = 0.15,
           ring_wall_mm: float = 5.0) -> dict:
    """Every number the parts need, or a refusal naming which rule broke and by how much."""
    m = float(module)
    s, p, n = int(sun), int(planet), int(n_planets)
    if m <= 0 or s < 4 or p < 4 or n < 2:
        return {"ok": False, "problems": ["a module, a sun, a planet and at least two planets"]}
    ring = s + 2 * p

    problems, warnings = [], []
    if (s + ring) % n:
        problems.append(
            f"{n} planets cannot sit at equal angles: (sun {s} + ring {ring}) = {s + ring}, "
            f"which is not divisible by {n}. Nearest that work: "
            + ", ".join(str(k) for k in range(2, 9) if (s + ring) % k == 0))

    centre = m * (s + p) / 2.0                       # sun axis to planet axis
    tip_r = m * (p + 2) / 2.0                        # planet tip radius (addendum = 1 module)
    if n >= 2:
        # Chord between neighbouring planet axes, against two tip radii plus air.
        chord = 2.0 * centre * math.sin(math.pi / n)
        gap = chord - 2.0 * tip_r
        if gap < PLANET_GAP_M * m:
            problems.append(
                f"neighbouring planets foul: their axes are {chord:.2f} mm apart and the tips need "
                f"{2 * tip_r:.2f} mm plus clearance, leaving {gap:.2f} mm. Fewer planets, a smaller "
                f"planet, or a bigger sun.")

    for label, z in (("sun", s), ("planet", p)):
        if z < MIN_TEETH:
            warnings.append(f"the {label} has {z} teeth; below {MIN_TEETH} at {pressure_angle:g} "
                            f"degrees the root undercuts and the flank is no longer a true involute")

    ratio = 1.0 + ring / s                           # carrier out, ring held
    ring_od = m * (ring + 2 * 1.25) + 2 * ring_wall_mm

    out = {
        "ok": not problems, "problems": problems, "warnings": warnings,
        "module": m, "pressure_angle": pressure_angle, "backlash_mm": backlash_mm,
        "sun_teeth": s, "planet_teeth": p, "ring_teeth": ring, "n_planets": n,
        "ratio": round(ratio, 4),
        "ratio_label": f"{ratio:.2f}:1 (carrier out, ring held)",
        "centre_distance_mm": round(centre, 3),
        "carrier_pin_circle_mm": round(2 * centre, 3),
        "sun_pitch_d_mm": round(m * s, 3), "planet_pitch_d_mm": round(m * p, 3),
        "ring_pitch_d_mm": round(m * ring, 3), "ring_od_mm": round(ring_od, 2),
        "face_mm": face_mm,
        "checks": [
            {"rule": "ring = sun + 2 x planet", "ok": True,
             "detail": f"{ring} = {s} + 2 x {p}", "why": "the planets have to span the gap exactly"},
            {"rule": "(sun + ring) divisible by planets", "ok": not (s + ring) % n,
             "detail": f"({s} + {ring}) % {n} = {(s + ring) % n}",
             "why": "otherwise they cannot be equally spaced"},
            {"rule": "planet tips clear each other", "ok": not any("foul" in x for x in problems),
             "detail": f"{2 * centre * math.sin(math.pi / n) - 2 * tip_r:.2f} mm between tips",
             "why": "adjacent planets turn in opposite directions at the rim"},
            {"rule": f"at least {MIN_TEETH} teeth", "ok": not warnings,
             "detail": f"sun {s}, planet {p}", "why": "fewer undercuts the root"},
        ],
    }
    return out


def suggest(module: float = 1.5, n_planets: int = 3, want_ratio: float = 4.0,
            max_ring_od_mm: float = 200.0) -> list[dict]:
    """Whole gearsets that satisfy every rule, closest to the ratio asked for first."""
    out = []
    for s in range(MIN_TEETH, 40):
        for p in range(MIN_TEETH, 40):
            d = design(module, s, p, n_planets)
            if not d["ok"] or d["warnings"] or d["ring_od_mm"] > max_ring_od_mm:
                continue
            out.append(d)
    out.sort(key=lambda d: (abs(d["ratio"] - want_ratio), d["ring_od_mm"]))
    return out[:8]


def _selfcheck() -> None:
    # the pick that looks right and is not: it meshes, and three planets cannot be equally spaced
    bad = design(1.5, 16, 16, 3)
    assert not bad["ok"], "16/16/48 with three planets must be refused"
    assert "not divisible by 3" in bad["problems"][0]
    assert "64" in bad["problems"][0], "the refusal has to carry the number"

    good = design(1.5, 18, 18, 3)
    assert good["ok"], good["problems"]
    assert good["ring_teeth"] == 54 and good["ratio"] == 4.0
    assert good["centre_distance_mm"] == 27.0
    assert good["carrier_pin_circle_mm"] == 54.0
    assert not good["warnings"], good["warnings"]

    # too many planets for the space they have to share
    crowded = design(1.5, 18, 18, 8)
    assert not crowded["ok"] and any("foul" in x for x in crowded["problems"])

    # a small sun is reported, not refused: it still runs, the flank is just not a true involute
    small = design(1.0, 12, 12, 3)
    assert small["ok"] and small["warnings"], "an undercut sun should warn rather than block"

    picks = suggest(1.5, 3, 4.0)
    assert picks and all(d["ok"] and not d["warnings"] for d in picks)
    assert abs(picks[0]["ratio"] - 4.0) < 0.01, picks[0]["ratio_label"]
    print(f"planetary ok — 16/16/48 x3 refused with its number, 18/18/54 x3 gives "
          f"{good['ratio_label']} on a {good['carrier_pin_circle_mm']} mm pin circle")


if __name__ == "__main__":
    _selfcheck()
