# Shared tile/grout layout recorded when a tiled shader is built.
# Tile chips read this back so they can sit on the same joints as the host.

from __future__ import annotations

import json

_LAST: dict = {}


def record_tile_layout(**kwargs) -> None:
    _LAST.clear()
    clean = {}
    for key, val in kwargs.items():
        if val is None:
            continue
        if hasattr(val, "tolist"):
            val = val.tolist()
        if isinstance(val, (tuple, list)):
            val = [float(x) for x in val]
        elif isinstance(val, (bool, int)):
            val = int(val)
        elif isinstance(val, float):
            val = float(val)
        else:
            val = str(val)
        clean[key] = val
    _LAST.update(clean)


def take_recorded_layout() -> dict:
    data = dict(_LAST)
    _LAST.clear()
    return data


def dump_layout(mat, layout: dict) -> None:
    if mat is None or not layout:
        return
    mat["syndefect_tile_layout"] = json.dumps(layout)
    mat["syndefect_jointed"] = True


def load_layout(mat) -> dict | None:
    if mat is None:
        return None
    raw = mat.get("syndefect_tile_layout")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def layout_from_object(obj) -> tuple[object, dict] | None:
    """Return (material, layout) for the first jointed material on ``obj``."""
    if obj is None:
        return None
    mats = []
    data = getattr(obj, "data", None)
    if data is not None:
        mats.extend(m for m in getattr(data, "materials", []) if m is not None)
    if getattr(obj, "active_material", None) is not None:
        mats.insert(0, obj.active_material)
    seen = set()
    for mat in mats:
        if mat is None or mat.name in seen:
            continue
        seen.add(mat.name)
        layout = load_layout(mat)
        if layout:
            return mat, layout
    return None
