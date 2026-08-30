# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE
# file in the root directory of this source tree.

"""One chrome finish shared by every bathroom fitting in a scene.

Taps, shower valves, rails and hooks in a real bathroom are bought as a set, so
they need to agree. Each factory used to sample its own metal - the sink tap in
particular drew from a list that could hand it a near-white finish - which left
the fittings looking like they came from four different suppliers.
"""

import bpy

from infinigen.assets.objects.wall_decorations.primitives import solid_material

_CHROME_NAME = "BathroomChrome"


def bathroom_chrome():
    """Polished chrome, created once per blend file and reused."""
    mat = bpy.data.materials.get(_CHROME_NAME)
    if mat is None:
        mat = solid_material(
            _CHROME_NAME,
            (0.74, 0.75, 0.77),
            roughness=0.10,
            metallic=1.0,
        )
        mat.name = _CHROME_NAME
    return mat
