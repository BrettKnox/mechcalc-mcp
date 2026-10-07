"""One material table. Every other table in the app is a view over this one.

There were five, and they disagreed about which materials existed. The damage was not that the numbers
differed (where they overlapped they matched) but that each table's `.get(name, DEFAULT)` silently
substituted a different material when a name was missing from it:

  * `fea.MATERIALS` had no asa/pc/carbon/copper/brass and fell back to `MATERIALS["aluminum"]`, so
    `fea.beam(material="asa")` analysed aluminium: 69000 MPa instead of 2000, a 34x stiffness error,
    reported as a clean pass.
  * `_common.E_of("brass")` returned PLA's 3500 MPa, a 33x error the other way.
  * `fea.thermal` defaulted a missing CTE to aluminium's 23, while `environment.fit_shift_um`
    defaulted the same missing value to PLA's 70.

So lookups here RAISE. A wrong number that looks right is worse than a stop, because the whole point of
the surrounding system is that its outputs can be trusted.

Two honest limitations, stated rather than hidden:

`knockdown` is the stand-in for a statistical A/B-basis allowable. Real allowables need ~100 coupons; a
hobbyist has none. 0.8 on printed polymers is a conventional derate covering layer adhesion, voids and
batch variation, not a computed lower bound, and `source` says so per row.

`z_strength_ratio` is the FDM anisotropy that decides whether a part survives: a printed part is far
weaker across layers than along them, so load must be oriented along them. Published ranges are wide
(Formlabs measured Z at ~55% of X on their tensile bars; other sources quote 40-75%), so these are
mid-range values to design against, not measurements of your printer.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


class UnknownMaterial(KeyError):
    """Raised instead of silently substituting a different material."""


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    family: str                 # polymer | metal | composite
    E_mpa: float
    yield_mpa: float | None     # None = brittle, no yield point; design against uts with a factor
    uts_mpa: float
    density_kgm3: float
    cte_ppm_c: float
    poisson: float
    max_service_c: float | None  # None = not the limiting factor at the temperatures a hobby build sees
    uv_stable: bool
    printable: bool             # can a printer in a hobby shop make it
    z_strength_ratio: float     # strength across layers / along layers (1.0 = isotropic stock)
    knockdown: float            # design derate applied to strength (see module docstring)
    source: str
    # BEARING is not tension. A steel pin in a printed bore crushes the bore long before the part
    # yields in tension, and the two allowables are different numbers -- `pin_bearing` was using the
    # tensile one, which is about twice as generous as the evidence supports.
    #
    # ASTM D5961 pin-bearing coupons in 100% infill PLA (Khosravani et al., Polymers 2023) give an
    # ULTIMATE bearing strength of 60 MPa at tight geometry (W/D 3, e/D 1.5) rising to 184 MPa at
    # very generous edge distance. 60 is the honest anchor for a compact lug, and PLA is brittle and
    # creeps, so a factor of 3 puts the design allowable at 20.
    bearing_mpa: float | None = None

    @property
    def is_metal(self) -> bool:
        return self.family == "metal"

    @property
    def bearing_allowable_mpa(self) -> float:
        """What a pin may press on. Falls back to the tensile allowable for metals, where the two
        are close enough and the bore is not the failure mode anyway."""
        return float(self.bearing_mpa) if self.bearing_mpa is not None else self.allowable_mpa

    @property
    def allowable_mpa(self) -> float:
        """The stress to design against: yield if it has one, else uts with a brittle factor, derated."""
        base = self.yield_mpa if self.yield_mpa is not None else self.uts_mpa / 2.0
        return round(base * self.knockdown, 1)


def _m(**kw) -> Material:
    return Material(**kw)


MATERIALS: dict[str, Material] = {m.key: m for m in [
    # ── printed polymers ──────────────────────────────────────────────────────
    _m(key="pla", name="PLA", family="polymer", E_mpa=3500, yield_mpa=50, uts_mpa=55,
       density_kgm3=1240, cte_ppm_c=70.0, poisson=0.36, max_service_c=50.0, uv_stable=False,
       # 0.40, not 0.50. Peer-reviewed Z/XY for a well-printed PLA coupon is 0.5-0.7 (Perez 2021
       # measures 0.52 of bulk), but the published range runs from 0.09 to ~1.0 and it is the print
       # that decides: >=0.3 mm layers, a nozzle under 200 C or hard part cooling drop it to 0.3 or
       # below. 0.5 is only defensible with the parameters pinned, and nothing here pins them.
       printable=True, z_strength_ratio=0.40, knockdown=0.8, bearing_mpa=20.0,
       source="Z/XY from Perez 2021 and the IJAMT 2025 spread; bearing from ASTM D5961 coupons "
              "(Khosravani 2023, 60 MPa ultimate at tight geometry) with a factor of 3"),
    _m(key="petg", name="PETG", family="polymer", E_mpa=2100, yield_mpa=50, uts_mpa=53,
       density_kgm3=1270, cte_ppm_c=60.0, poisson=0.38, max_service_c=70.0, uv_stable=False,
       printable=True, z_strength_ratio=0.55, knockdown=0.8, source="typical FDM datasheet values"),
    _m(key="abs", name="ABS", family="polymer", E_mpa=2200, yield_mpa=40, uts_mpa=43,
       density_kgm3=1050, cte_ppm_c=90.0, poisson=0.35, max_service_c=85.0, uv_stable=False,
       printable=True, z_strength_ratio=0.55, knockdown=0.8,
       source="typical FDM datasheet values; chalks under UV"),
    _m(key="asa", name="ASA", family="polymer", E_mpa=2000, yield_mpa=42, uts_mpa=45,
       density_kgm3=1070, cte_ppm_c=85.0, poisson=0.35, max_service_c=90.0, uv_stable=True,
       printable=True, z_strength_ratio=0.55, knockdown=0.8,
       source="typical FDM datasheet values; the UV-stable ABS replacement"),
    _m(key="pc", name="Polycarbonate", family="polymer", E_mpa=2300, yield_mpa=60, uts_mpa=65,
       density_kgm3=1200, cte_ppm_c=65.0, poisson=0.37, max_service_c=110.0, uv_stable=True,
       printable=True, z_strength_ratio=0.60, knockdown=0.8,
       source="typical FDM datasheet values; needs a hot end above 260 C and an enclosure"),
    _m(key="nylon", name="Nylon (PA)", family="polymer", E_mpa=2000, yield_mpa=45, uts_mpa=50,
       density_kgm3=1140, cte_ppm_c=80.0, poisson=0.39, max_service_c=95.0, uv_stable=False,
       printable=True, z_strength_ratio=0.60, knockdown=0.8,
       source="typical FDM/SLS values; hygroscopic, properties drop when wet"),
    # ── stock metals (bought, not printed) ────────────────────────────────────
    _m(key="aluminum", name="Aluminium 6061-T6", family="metal", E_mpa=69000, yield_mpa=240,
       uts_mpa=290, density_kgm3=2700, cte_ppm_c=23.0, poisson=0.33, max_service_c=200.0,
       uv_stable=True, printable=False, z_strength_ratio=1.0, knockdown=1.0,
       source="6061-T6 handbook values"),
    _m(key="steel", name="Mild steel", family="metal", E_mpa=210000, yield_mpa=250, uts_mpa=400,
       density_kgm3=7850, cte_ppm_c=12.0, poisson=0.30, max_service_c=400.0, uv_stable=True,
       printable=False, z_strength_ratio=1.0, knockdown=1.0,
       source="mild/low-carbon steel handbook values"),
    _m(key="titanium", name="Titanium Ti-6Al-4V", family="metal", E_mpa=114000, yield_mpa=830,
       uts_mpa=900, density_kgm3=4500, cte_ppm_c=8.6, poisson=0.34, max_service_c=400.0,
       uv_stable=True, printable=False, z_strength_ratio=1.0, knockdown=1.0,
       source="Ti-6Al-4V handbook values"),
    _m(key="copper", name="Copper", family="metal", E_mpa=117000, yield_mpa=70, uts_mpa=220,
       density_kgm3=8960, cte_ppm_c=17.0, poisson=0.34, max_service_c=200.0, uv_stable=True,
       printable=False, z_strength_ratio=1.0, knockdown=1.0,
       source="annealed copper; work-hardened runs far higher"),
    _m(key="brass", name="Brass", family="metal", E_mpa=100000, yield_mpa=200, uts_mpa=350,
       density_kgm3=8500, cte_ppm_c=19.0, poisson=0.34, max_service_c=200.0, uv_stable=True,
       printable=False, z_strength_ratio=1.0, knockdown=1.0, source="C36000 free-cutting brass"),
    # ── composite ─────────────────────────────────────────────────────────────
    _m(key="carbon", name="Carbon fibre (pultruded)", family="composite", E_mpa=70000,
       yield_mpa=None, uts_mpa=600, density_kgm3=1600, cte_ppm_c=2.0, poisson=0.30,
       max_service_c=100.0, uv_stable=False, printable=False, z_strength_ratio=0.15, knockdown=0.6,
       source="generic pultruded tube/rod, fibres along the axis. Highly anisotropic: transverse and "
              "interlaminar strength are a small fraction of axial, and it fails brittle with no "
              "yield warning. Treat as indicative only; a real layup needs its own numbers."),
]}

# Names seen in the wild -> canonical key. Substring-matched too, so "6061-T6 aluminium extrusion" maps.
ALIASES = {
    "aluminium": "aluminum", "al": "aluminum", "6061": "aluminum", "6061-t6": "aluminum",
    "7075": "aluminum", "alu": "aluminum",
    "mild steel": "steel", "stainless": "steel", "stainless steel": "steel", "304": "steel",
    "mild-steel": "steel", "ti": "titanium", "ti-6al-4v": "titanium",
    "pa": "nylon", "pa12": "nylon", "pa6": "nylon", "nylon12": "nylon",
    "pla+": "pla", "pla plus": "pla", "polylactic acid": "pla",
    "polycarbonate": "pc", "makrolon": "pc",
    "carbon fiber": "carbon", "carbon fibre": "carbon", "cf": "carbon", "cfrp": "carbon",
}


def canonical(name: str) -> str | None:
    """Map a free-text material name onto a table key, or None."""
    low = str(name or "").strip().lower()
    if not low:
        return None
    if low in MATERIALS:
        return low
    if low in ALIASES:
        return ALIASES[low]
    for alias, key in ALIASES.items():          # "6061-T6 aluminium extrusion"
        if alias in low:
            return key
    for key in MATERIALS:
        if key in low:
            return key
    return None


# ── measured overrides ────────────────────────────────────────────────────────────────────────────
# A datasheet describes moulded resin; your part is a stack of extrusions. Once calibrate.py has
# real coupon data for a material, it REPLACES the table value here rather than sitting in a report
# nobody reads. `source` is rewritten too, so any number downstream can say where it came from.
# Cached against the file's mtime because get() is called inside the optimiser's inner loop.
_overlay_cache: tuple[float, dict] = (-1.0, {})


def _overlay() -> dict:
    global _overlay_cache
    from . import calibrate
    try:
        mtime = calibrate.STORE.stat().st_mtime
    except OSError:
        _overlay_cache = (-1.0, {})
        return {}
    if mtime != _overlay_cache[0]:
        try:
            _overlay_cache = (mtime, calibrate.overlay())
        except Exception:      # noqa: BLE001 - a corrupt calibration file must not break every lookup
            _overlay_cache = (mtime, {})
    return _overlay_cache[1]


def get(name: str) -> Material:
    """The material, or a loud failure. Never a silent substitute.

    Measured data for this material wins over the table. That is the whole point of measuring it.
    """
    key = canonical(name)
    if key is None:
        raise UnknownMaterial(
            f"unknown material {name!r}. Known: {', '.join(sorted(MATERIALS))}. "
            f"Add it to matprops.py rather than letting a lookup substitute another material.")
    base = MATERIALS[key]
    patch = _overlay().get(key)
    return replace(base, **patch) if patch else base


def is_measured(name: str) -> bool:
    """True when this material's numbers came off your own printer rather than a datasheet."""
    key = canonical(name)
    return bool(key and key in _overlay())


def has(name: str) -> bool:
    return canonical(name) is not None


def keys() -> list[str]:
    return sorted(MATERIALS)


def printable() -> list[str]:
    return sorted(k for k, m in MATERIALS.items() if m.printable)


def table(attr: str) -> dict:
    """A legacy-shaped dict view, so the old module-level tables stay importable."""
    return {k: getattr(m, attr) for k, m in MATERIALS.items()}


def _selfcheck() -> None:
    # the exact substitutions that used to happen silently
    assert get("asa").E_mpa == 2000, "ASA must not resolve to aluminium's 69000"
    assert get("brass").E_mpa == 100000, "brass must not resolve to PLA's 3500"
    assert get("carbon").cte_ppm_c == 2.0, "carbon had no CTE anywhere; it must not default to 23 or 70"
    for bad in ("unobtainium", "", "wood", None):
        try:
            get(bad)
        except UnknownMaterial:
            pass
        else:
            raise AssertionError(f"{bad!r} must raise, not substitute")

    # aliases people actually type
    assert canonical("6061-T6 aluminium extrusion") == "aluminum"
    assert canonical("PLA+") == "pla" and canonical("PA12") == "nylon"
    assert canonical("Carbon Fibre") == "carbon"

    # every material must be self-consistent
    for k, m in MATERIALS.items():
        assert m.uts_mpa > 0 and m.E_mpa > 0 and m.density_kgm3 > 0, k
        if m.yield_mpa is not None:
            assert m.yield_mpa <= m.uts_mpa, f"{k}: yield above uts"
        assert 0 < m.z_strength_ratio <= 1.0, k
        assert 0 < m.knockdown <= 1.0, k
        assert m.source, f"{k}: every number needs a stated basis"
        assert m.printable == (m.family == "polymer"), k

    # the anisotropy that decides part orientation
    assert MATERIALS["pla"].z_strength_ratio < 0.6, "printed PLA is much weaker across layers"
    assert MATERIALS["steel"].z_strength_ratio == 1.0, "stock metal is isotropic"
    # brittle materials have no yield, so the allowable comes off uts with a factor
    assert MATERIALS["carbon"].yield_mpa is None
    assert MATERIALS["carbon"].allowable_mpa == round(600 / 2 * 0.6, 1)
    assert MATERIALS["pla"].allowable_mpa == 40.0                      # 50 * 0.8
    assert MATERIALS["steel"].allowable_mpa == 250.0                   # no derate on stock metal
    print(f"matprops ok — {len(MATERIALS)} materials, {len(printable())} printable; "
          f"ASA E={get('asa').E_mpa} (was silently 69000), brass E={get('brass').E_mpa} "
          f"(was silently 3500); unknown names raise")


if __name__ == "__main__":
    _selfcheck()
