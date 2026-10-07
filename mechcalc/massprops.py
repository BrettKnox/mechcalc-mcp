"""massprops.py — mass, centre of gravity and moment of inertia from a real component layout.

This is the piece the repo never had: every engine took `mass_kg` as one guessed scalar, and nothing
computed a CoG or an inertia anywhere. Given a build configuration (which parts, how many, and where),
this returns the aggregate physical properties the engines actually need.

Coordinate frame (load-bearing — everything downstream assumes it):
    right-handed, millimetres, origin at the GROUND CONTACT PLANE, mid-wheelbase, on the centreline.
    +x forward (= travel direction, = aero.frontal_area_m2(axis="x")), +y left, +z up.
    So pos_mm[2] IS height above ground and CoG height needs no offset arithmetic.
    For arm/cnc/humanoid the origin is the base mount face centre, +z up.
`pos_mm` is a component's centre of mass: the point-mass approximation, stated openly. A solid-cuboid
self-inertia term is added when a part's bounding box is known, which removes the biggest bias of a
pure point-mass model (a 90 mm plate at a 20 mm offset carries most of its inertia in itself).

Pure functions, no I/O, no numpy — this runs in the configuration-search hot loop.
"""
from __future__ import annotations

G = 9.81


def battery_mass_g(capacity_mah: float, v_nom: float, wh_per_kg: float = 150.0) -> float:
    """Pack mass from energy — extracted verbatim from engines/drone.py:30 so every engine estimates
    it identically. Only used when the chosen battery Component has no measured mass_g."""
    try:
        wh = (float(capacity_mah) / 1000.0) * float(v_nom)
        return wh / float(wh_per_kg) * 1000.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _dims_mm(c: dict) -> tuple[float, float, float] | None:
    """Bounding box in mm, from l/w/h, or from a diameter plus whatever length is recorded."""
    if not c:
        return None
    l, w, h = _f(c.get("len_mm")), _f(c.get("wid_mm")), _f(c.get("hgt_mm"))
    if l and w and h:
        return (l, w, h)
    # The OUTSIDE is what occupies space. `dia_mm` on a bearing is its BORE -- taking it first drew
    # a 6200 as a 10 mm cube when it is 30 across and 9 thick.
    d = _f(c.get("od_mm")) or _f(c.get("dia_mm"))
    if d:
        # A ROD IS NOT A CUBE either. With a length recorded, use it: a 400 mm leadscrew and a
        # 300 mm shaft were both drawing as 8 mm cubes, which hides every clash they could have.
        return (d, d, _f(c.get("hgt_mm")) or _f(c.get("wid_mm")) or l or d)
    return None


def _measured(c: dict, key: str):
    """A measured value for this spec, if someone actually put a caliper on it."""
    m = c.get("measured")
    if isinstance(m, str):
        import json as _j
        try:
            m = _j.loads(m)
        except (TypeError, ValueError):
            m = None
    if isinstance(m, dict):
        v = m.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _mass_g(c: dict) -> float:
    """A component's mass: MEASURED if someone weighed it, else the datasheet, else an estimate.

    Preferring the measured value is the single line that closes the re-optimise-on-measured loop. A
    datasheet mass is an optimistic average; a printed bracket and a bag of hardware come out heavy, and
    mass propagates into every other number through CoG and inertia.
    """
    meas = _measured(c, "mass_g")
    if meas:
        return meas
    m = _f(c.get("mass_g"))
    if m:
        return m
    if c.get("capacity_mah"):
        v = _f(c.get("voltage_nom_v")) or (_f(c.get("cells")) * 3.7)
        return battery_mass_g(c["capacity_mah"], v or 3.7)
    return 0.0


def resolve(cfg: dict, components: dict, slot_defaults: dict | None = None) -> list[dict]:
    """Flatten a config's slots into one entry per physical instance.

    Expands `qty` (mirroring or corner-placing when a slot has no explicit position), applies per-slot
    `overrides`, and honours a bare `mass_g` on a slot with no component (payload, wiring, fasteners).
    Returns [{slot, category, name, mass_kg, pos_mm:(x,y,z), dims_mm|None, spec}].

    `slot_defaults` maps a slot name to what the BUILD TYPE says about it -- where the component
    mounts, which way a rod runs, what length it is cut to -- used where the saved config has no
    opinion. Without it every bought part resolves to the origin: eleven of them stacked inside the
    excavator's turntable, contributing a meaningless centre of gravity and excluded from the
    assembly as unplaceable. The config still wins on anything it does pin.
    """
    out: list[dict] = []
    layout = (cfg or {}).get("layout") or {}
    wb = _f(layout.get("wheelbase_mm"), 0.0)
    track = _f(layout.get("track_mm"), 0.0)
    for item in (cfg or {}).get("items") or []:
        # The build type underneath, the config on top -- but only where the config actually SAYS
        # something. `blank()` writes `pos_mm: None` into every item, and a plain dict merge lets
        # that None overwrite the declared position, which put every bought part back at the origin
        # with the constraint apparently in place.
        slot = {**(slot_defaults or {}).get(item.get("slot"), {}),
                **{k: v for k, v in item.items() if v is not None}}
        cid = slot.get("component_id")
        comp = dict(components.get(cid) or {}) if cid is not None else {}
        comp.update(slot.get("overrides") or {})
        qty = max(1, int(_f(slot.get("qty") or slot.get("count") or 1, 1)))
        each_g = _mass_g(comp) if comp else _f(slot.get("mass_g"))
        dims = _dims_mm(comp)
        # STOCK IS NOT INSTALLED LENGTH. You buy a 400 mm leadscrew and a 300 mm shaft and cut them;
        # drawn at stock the screw ran clean past the bucket and the pivot shaft stuck 150 mm out of
        # both sides of the machine. `cut_to_mm` is what actually goes in, and for the screw it is a
        # real design number -- an uncut stick buckles below the force its own motor makes.
        cut = _f(slot.get("cut_to_mm"))
        if cut and dims:
            dims = (dims[0], dims[1], cut)
        base = slot.get("pos_mm")
        for i in range(qty):
            pos = _spread(base, i, qty, wb, track, comp, slot)
            out.append({"slot": slot.get("slot") or slot.get("role") or comp.get("category") or "part",
                        "category": comp.get("category") or slot.get("category"),
                        "name": comp.get("name") or slot.get("role") or "mass",
                        "mass_kg": each_g / 1000.0, "pos_mm": pos, "dims_mm": dims,
                        "spec": comp, "component_id": cid})
    return out


def _spread(base, i: int, qty: int, wheelbase_mm: float, track_mm: float,
            comp: dict, slot: dict) -> tuple[float, float, float] | None:
    """Where instance i of a qty-n slot sits, or None when this instance has nowhere to go.

    An explicit pos is used as-is for the first instance and mirrored across ±y for pairs; wheels
    with no pos land on the wheelbase/track corners. None is a real answer -- see the multiple-of-
    the-pattern rule below -- and every caller has to carry it through rather than defaulting.
    """
    # A slot may declare ONE position or a list of them. Three joint motors do not live in one
    # place -- slew, boom ram and stick ram are three different mounts -- and a single position
    # stacked all three inside each other.
    if base and len(base) and isinstance(base[0], (list, tuple)):
        picks = [b for b in base if b and len(b) >= 3]
        if picks:
            # CYCLE, do not clamp. `min(i, len-1)` put instances past the last declared position all
            # on that position: the tool coupler declares four bolt-circle points and the tree buys
            # eight bolts -- four for the bucket and four for the gripper -- so bolts 4 through 7
            # landed on top of bolt 3. Repeats of a fixed pattern are the normal reason a qty exceeds
            # the positions, and cycling places the second set on the same circle as the first.
            #
            # ONLY WHEN IT REALLY IS A REPEAT. Eight bolts on four points is two sets; THREE motors
            # on two mounts is not, and cycling put the third NEMA 17 inside the first -- two 66 mm
            # steppers in one 66 mm volume, which `clashes` then reported as no clash at all. A
            # count that is not a whole multiple of the pattern has an instance with nowhere to go,
            # and this codebase's rule for that is to say so, not to stack it on something.
            if qty % len(picks):
                return None if i >= len(picks) else (_f(picks[i][0]), _f(picks[i][1]), _f(picks[i][2]))
            b = picks[i % len(picks)]
            return (_f(b[0]), _f(b[1]), _f(b[2]))
    if base and len(base) >= 3 and any(base):
        x, y, z = _f(base[0]), _f(base[1]), _f(base[2])
        if qty == 2:                                   # a pair straddles the centreline
            return (x, y if y else (track_mm / 2.0 if i == 0 else -track_mm / 2.0), z)
        if qty == 4 and track_mm and wheelbase_mm:
            sx = 1 if i in (0, 1) else -1
            sy = 1 if i in (0, 2) else -1
            return (x + sx * wheelbase_mm / 2.0 if not x else x, sy * track_mm / 2.0, z)
        return (x, y, z)
    # no explicit position: corner-place wheels, else sit on the centreline at half the part's height
    d = _dims_mm(comp)
    z = (_f(comp.get("dia_mm")) / 2.0) if comp.get("dia_mm") else ((d[2] / 2.0) if d else 20.0)
    cat = (comp.get("category") or slot.get("category") or "").lower()
    if cat == "wheel" and qty >= 2 and wheelbase_mm and track_mm:
        sx = 1 if i in (0, 1) else -1
        sy = 1 if i in (0, 2) else -1
        return (sx * wheelbase_mm / 2.0, sy * track_mm / 2.0, z)
    if qty == 2 and track_mm:
        return (0.0, (1 if i == 0 else -1) * track_mm / 2.0, z)
    return (0.0, 0.0, z)


def aggregate(items: list[dict]) -> dict:
    """Total mass, CoG and diagonal inertia about the CoG.

        M   = Σ mᵢ
        cog = Σ mᵢ rᵢ / M
        Ixx = Σ [ mᵢ((yᵢ−c_y)² + (zᵢ−c_z)²) + mᵢ/12·(bᵢ² + cᵢ²) ]   ← 2nd term only when dims known

    Diagonal only (Ixx/Iyy/Izz) — nothing in the app consumes products of inertia.
    Lengths in metres internally; returns mm for CoG so it maps straight onto the engines' *_mm keys.
    """
    # A PART WITH NO POSITION HAS MASS AND NO MOMENT. It cannot go into a centre of gravity, and it
    # must not quietly go out of the total either -- it is still bought, still carried, and still
    # weighs what it weighs. So it is held out of the CoG arithmetic and reported by name, the same
    # way `simmanifest` keeps an unplaced body out of `bodies` rather than dropping it at the origin.
    lost = [i for i in items or [] if i.get("pos_mm") is None]
    items = [i for i in items or [] if i.get("pos_mm") is not None]
    lost_kg = sum(max(0.0, i.get("mass_kg") or 0.0) for i in lost)
    unplaced = {"unplaced_kg": round(lost_kg, 4),
                "unplaced": [{"slot": i.get("slot"), "name": i.get("name")} for i in lost],
                "why_unplaced": ("these have mass but no declared position, so they are excluded "
                                 "from the centre of gravity and NOT from the machine")} if lost else {}
    M = sum(max(0.0, i.get("mass_kg") or 0.0) for i in items)
    if M <= 0:
        return {"total_mass_kg": 0.0, "cog_mm": (0.0, 0.0, 0.0), "inertia_kgm2": (0.0, 0.0, 0.0),
                "by_category": {}, "count": len(items), **unplaced}
    cx = sum((i["mass_kg"] or 0.0) * i["pos_mm"][0] for i in items) / M
    cy = sum((i["mass_kg"] or 0.0) * i["pos_mm"][1] for i in items) / M
    cz = sum((i["mass_kg"] or 0.0) * i["pos_mm"][2] for i in items) / M
    ixx = iyy = izz = 0.0
    for it in items:
        m = it.get("mass_kg") or 0.0
        if m <= 0:
            continue
        dx = (it["pos_mm"][0] - cx) / 1000.0
        dy = (it["pos_mm"][1] - cy) / 1000.0
        dz = (it["pos_mm"][2] - cz) / 1000.0
        ixx += m * (dy * dy + dz * dz)
        iyy += m * (dx * dx + dz * dz)
        izz += m * (dx * dx + dy * dy)
        d = it.get("dims_mm")
        if d:                                          # solid-cuboid self inertia
            a, b, c = (d[0] / 1000.0, d[1] / 1000.0, d[2] / 1000.0)
            ixx += m / 12.0 * (b * b + c * c)
            iyy += m / 12.0 * (a * a + c * c)
            izz += m / 12.0 * (a * a + b * b)
    by_cat: dict = {}
    for it in items:
        k = (it.get("category") or it.get("slot") or "other")
        by_cat[k] = round(by_cat.get(k, 0.0) + (it.get("mass_kg") or 0.0), 5)
    return {"total_mass_kg": round(M, 4),
            "cog_mm": (round(cx, 2), round(cy, 2), round(cz, 2)),
            "inertia_kgm2": (round(ixx, 6), round(iyy, 6), round(izz, 6)),
            "by_category": by_cat, "count": len(items), **unplaced}


def shell_mass_kg(layout: dict, material: str = "pla", infill: float = 0.35) -> float:
    """Printed-structure mass: shell surface × wall × density × an infill/rib allowance.
    Deliberately crude — it's a printed hollow body, not a solid — but it stops the chassis weighing 0."""
    from .engines._common import rho_of
    L = _f(layout.get("shell_l_mm")) or _f(layout.get("wheelbase_mm")) * 1.15
    W = _f(layout.get("shell_w_mm")) or _f(layout.get("track_mm")) * 1.05
    H = _f(layout.get("shell_h_mm"), 50.0)
    wall = _f(layout.get("wall_mm"), 1.6)
    if not (L and W and H and wall):
        return 0.0
    area_mm2 = 2 * (L * W) + 2 * (L * H) + 2 * (W * H)          # closed box surface
    vol_mm3 = area_mm2 * wall * (1.0 + infill)
    return round(vol_mm3 / 1e9 * rho_of(material), 4)           # mm³→m³ × kg/m³


if __name__ == "__main__":  # smoke: known two-mass layout, by hand
    items = [{"slot": "a", "mass_kg": 1.0, "pos_mm": (100.0, 0.0, 0.0), "dims_mm": None},
             {"slot": "b", "mass_kg": 1.0, "pos_mm": (-100.0, 0.0, 0.0), "dims_mm": None}]
    a = aggregate(items)
    assert a["total_mass_kg"] == 2.0, a
    assert a["cog_mm"] == (0.0, 0.0, 0.0), a                    # symmetric → CoG at origin
    # Izz for two 1 kg at ±0.1 m = 2 × 1 × 0.1² = 0.02 kg·m²
    assert abs(a["inertia_kgm2"][2] - 0.02) < 1e-9, a
    assert abs(a["inertia_kgm2"][0]) < 1e-9, a                  # both on the x axis → Ixx = 0
    # asymmetric: 3 kg at x=0, 1 kg at x=400 → CoG at 100 mm
    a2 = aggregate([{"slot": "a", "mass_kg": 3.0, "pos_mm": (0.0, 0.0, 0.0), "dims_mm": None},
                    {"slot": "b", "mass_kg": 1.0, "pos_mm": (400.0, 0.0, 0.0), "dims_mm": None}])
    assert a2["cog_mm"][0] == 100.0, a2
    # self-inertia term engages with dims
    solo = aggregate([{"slot": "p", "mass_kg": 1.2, "pos_mm": (0.0, 0.0, 0.0), "dims_mm": (100.0, 100.0, 10.0)}])
    assert solo["inertia_kgm2"][2] > 0, solo                    # a point mass alone would be 0
    print("massprops ok — CoG, parallel-axis and self-inertia all check out:", a2["cog_mm"], solo["inertia_kgm2"])
