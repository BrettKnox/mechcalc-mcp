"""engines — deterministic first-principles domain physics engines. Each module exposes the same trio
as analysis.py (DEFAULTS + analyze + plot). ENGINES maps an engine name to its module.
"""
from __future__ import annotations

from . import arm, balance, cnc, drone, humanoid, legged, linear, vacuum
from .. import analysis as _vehicle

ENGINES = {
    "vehicle": _vehicle,
    "arm": arm,
    "cnc": cnc,
    "multirotor": drone,
    "legged": legged,
    "humanoid": humanoid,
    "linear": linear,
    "balance": balance,
    "vacuum": vacuum,
}


def get(name: str):
    return ENGINES.get(name)
