"""fea.py: closed-form structural checks (no meshing). Beam, Euler column, bolt preload, thermal
expansion. Exact formulas, so the answer does not depend on a model getting it right. Units: mm, N, MPa.

Every result carries `assumptions`: the cases each formula does not cover. For arbitrary 3D parts use a
real FE solver (CalculiX, FreeCAD FEM) on the exported mesh.
"""
from __future__ import annotations

import math

from . import matprops as _mp

SUPPORTS = ("cantilever_tip", "simply_supported_center")
END_K = {"pinned": 1.0, "fixed_free": 2.0, "fixed_fixed": 0.5, "fixed_pinned": 0.7}


def _props(material: str):
    """(E, allowable stress). Raises UnknownMaterial rather than substituting another material."""
    m = _mp.get(material)
    return float(m.E_mpa), float(m.allowable_mpa)


def _positive(**kw) -> str | None:
    for k, v in kw.items():
        try:
            if not float(v) > 0:
                return f"{k} must be > 0, got {v!r}"
        except (TypeError, ValueError):
            return f"{k} must be a number, got {v!r}"
    return None


def beam(length: float, width: float, height: float, load: float, material: str = "aluminum",
         support: str = "cantilever_tip") -> dict:
    """Rectangular beam under a point load. cantilever_tip or simply_supported_center."""
    if err := _positive(length=length, width=width, height=height):
        return {"ok": False, "error": err}
    if support not in SUPPORTS:
        return {"ok": False, "error": f"support must be one of {SUPPORTS}"}
    try:
        E, sy = _props(material)
    except _mp.UnknownMaterial as e:
        return {"ok": False, "error": str(e)}
    L, b, h, P = float(length), float(width), float(height), float(load)
    I = b * h ** 3 / 12.0
    if support == "cantilever_tip":
        Mmax, dmax = P * L, P * L ** 3 / (3 * E * I)
    else:
        Mmax, dmax = P * L / 4, P * L ** 3 / (48 * E * I)
    smax = abs(Mmax) * (h / 2) / I
    sf = sy / smax if smax > 0 else math.inf
    return {"ok": True, "material": material, "support": support,
            "max_stress_mpa": round(smax, 3), "max_deflection_mm": round(abs(dmax), 5),
            "safety_factor": round(sf, 3), "second_moment_mm4": round(I, 2),
            "verdict": "OK" if sf >= 1 else "YIELDS", "E_mpa": E, "allowable_mpa": sy,
            "assumptions": ["rectangular section, linear elastic, small deflection",
                            "point load only; distributed loads are not modelled",
                            "allowable already includes the material knockdown in matprops"]}


def column(length: float, width: float, height: float, material: str = "aluminum",
           end: str = "pinned") -> dict:
    """Euler buckling vs squash yield for a rectangular column. end: pinned, fixed_free, fixed_fixed,
    fixed_pinned."""
    if err := _positive(length=length, width=width, height=height):
        return {"ok": False, "error": err}
    if end not in END_K:
        return {"ok": False, "error": f"end must be one of {sorted(END_K)}"}
    try:
        E, sy = _props(material)
    except _mp.UnknownMaterial as e:
        return {"ok": False, "error": str(e)}
    L, b, h, K = float(length), float(width), float(height), END_K[end]
    A = b * h
    Imin = min(b * h ** 3, h * b ** 3) / 12.0
    pcr = math.pi ** 2 * E * Imin / (K * L) ** 2
    pyield = A * sy
    return {"ok": True, "material": material, "end": end,
            "critical_load_N": round(min(pcr, pyield), 1), "euler_buckling_N": round(pcr, 1),
            "yield_load_N": round(pyield, 1), "governs": "buckling" if pcr < pyield else "yield",
            "slenderness": round(K * L / math.sqrt(Imin / A), 1),
            "assumptions": ["perfectly straight, centrally loaded column",
                            "real columns have initial curvature and eccentricity: Pcr is an upper bound"]}


def bolt(torque_nm: float, diameter_mm: float, k: float = 0.2) -> dict:
    """Preload from tightening torque, F = T/(K d), and the resulting bolt tensile stress."""
    if err := _positive(torque_nm=torque_nm, diameter_mm=diameter_mm, k=k):
        return {"ok": False, "error": err}
    T, d, K = float(torque_nm), float(diameter_mm), float(k)
    F = T * 1000.0 / (K * d)
    As = 0.7 * math.pi / 4 * d ** 2
    return {"ok": True, "clamp_force_N": round(F, 1), "bolt_tensile_stress_mpa": round(F / As, 2),
            "assumptions": ["tensile stress area approximated as 0.7*pi/4*d^2",
                            "K is nominal: about 0.2 dry, 0.1 lubricated; torque-only preload scatters +-25%"]}


def thermal(length_mm: float, delta_t: float, material: str = "aluminum", constrained: bool = False) -> dict:
    """Free expansion dL = a L dT, and the stress E a dT if fully restrained."""
    if err := _positive(length_mm=length_mm):
        return {"ok": False, "error": err}
    try:
        E, _ = _props(material)
        a = float(_mp.get(material).cte_ppm_c)
    except _mp.UnknownMaterial as e:
        return {"ok": False, "error": str(e)}
    dT = float(delta_t)
    return {"ok": True, "material": material, "expansion_mm": round(a * 1e-6 * float(length_mm) * dT, 5),
            "thermal_stress_mpa": round(E * a * 1e-6 * dT, 3) if constrained else 0.0,
            "cte_ppm_c": a, "E_mpa": E,
            "assumptions": ["constrained means FULL restraint, the worst case",
                            "linear CTE; polymers are not linear near their glass transition"]}


if __name__ == "__main__":
    # textbook cantilever: 6061, 100 x 10 x 10 mm, 100 N tip -> sigma = 6 P L / (b h^2) = 60 MPa
    r = beam(100, 10, 10, 100, "aluminum")
    assert abs(r["max_stress_mpa"] - 60.0) < 1e-6, r
    assert abs(bolt(1.31, 3)["clamp_force_N"] - 2183.3) < 0.1
    assert column(500, 5, 5, "aluminum", "pinned")["governs"] == "buckling"
    assert beam(-1, 1, 1, 1)["ok"] is False and column(1, 1, 1, end="glued")["ok"] is False
    print("fea self-check ok")
