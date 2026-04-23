#!/usr/bin/env python3
"""Count camera rig empties in a saved Infinigen scene.

Run with Blender (the .blend must be the first file argument so it is loaded
before this script runs)::

    blender --background path/to/scene.blend --python tools/count_camera_rigs.py

Prints a single integer (number of objects in the ``camera_rigs`` collection)
to stdout. Blender 4+ may print extra lines after this; when parsing in shell,
use ``grep -E '^[0-9]+$' | head -n1``, not ``tail -n1``.

This matches :func:`infinigen.core.placement.camera.get_camera_rigs` (one rig
empty ``camrig.{i}`` per entry).
"""

from __future__ import annotations

import bpy


def main() -> None:
    col = bpy.data.collections.get("camera_rigs")
    if col is None:
        print(0)
        return
    print(len(col.objects))


if __name__ == "__main__":
    main()
