# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE
# file in the root directory of this source tree.

"""Casement window handle: backplate plus a cranked lever, as on flat glazing."""

from numpy.random import uniform

from infinigen.assets.objects.wall_decorations.primitives import (
    assign,
    rounded_box,
    shade_smooth,
)
from infinigen.assets.utils.object import join_objects
from infinigen.core import surface
from infinigen.core.util.math import FixedSeed


def make_window_handle(seed, material):
    """Build one handle at the origin, protruding along +X, lever hanging in -Z.

    The caller orients and positions it; ``material`` is shared with the door
    furniture so all the ironmongery in a flat reads as the same metal.
    """
    with FixedSeed(seed):
        plate_w = uniform(0.026, 0.034)
        plate_h = uniform(0.085, 0.115)
        plate_d = uniform(0.005, 0.008)
        lever_len = uniform(0.075, 0.105)
        lever_r = uniform(0.007, 0.010)
        stem = uniform(0.016, 0.024)

    plate = rounded_box(
        (plate_d, plate_w, plate_h),
        location=(plate_d / 2, 0, 0),
        radius=min(0.004, plate_d * 0.45),
        segments=4,
        name="windowhandle_plate",
    )
    parts = [plate]

    # Stem stands off the plate, then the lever runs down across the sash.
    neck = rounded_box(
        (stem, lever_r * 2.1, lever_r * 2.1),
        location=(plate_d + stem / 2, 0, 0),
        radius=lever_r * 0.6,
        segments=4,
        name="windowhandle_neck",
    )
    parts.append(neck)

    lever = rounded_box(
        (lever_r * 1.8, lever_r * 1.9, lever_len),
        location=(
            plate_d + stem - lever_r * 0.3,
            0,
            -lever_len / 2 + lever_r,
        ),
        radius=lever_r * 0.75,
        segments=5,
        name="windowhandle_lever",
    )
    parts.append(lever)

    obj = join_objects(parts)
    shade_smooth(obj)
    if material is not None:
        surface.assign_material(obj, material)
    obj.name = f"WindowHandle_{seed}"
    return obj
