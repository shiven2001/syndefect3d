"""Deck-mounted kitchen mixer (chrome, single-lever or two-handle)."""

import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.wall_decorations.primitives import (
    assign,
    box,
    cylinder,
    shade_smooth,
    solid_material,
)
from infinigen.assets.utils.object import join_objects, new_bbox
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class KitchenMixerFactory(AssetFactory):
    """Sits on the sink deck; spout points +X into the basin."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.style = "lever" if uniform() < 0.7 else "two_handle"
            self.rise = uniform(0.22, 0.30)
            self.reach = uniform(0.16, 0.22)
            self.pipe_r = uniform(0.010, 0.014)
            self.base_r = uniform(0.022, 0.030)

    def create_placeholder(self, **params):
        return new_bbox(
            -self.base_r,
            self.reach + 0.03,
            -0.07,
            0.07,
            0,
            self.rise + 0.04,
        )

    def create_asset(self, **params):
        chrome = solid_material(
            f"KitchenMixer_{self.factory_seed}",
            (0.74, 0.75, 0.77),
            roughness=0.10,
            metallic=1.0,
        )
        parts = []
        base = cylinder(self.base_r, 0.014, location=(0, 0, 0.007), name="kmix_base")
        assign(base, chrome)
        shade_smooth(base)
        parts.append(base)
        riser = cylinder(
            self.pipe_r,
            self.rise * 0.72,
            location=(0, 0, 0.014 + self.rise * 0.36),
            name="kmix_riser",
        )
        assign(riser, chrome)
        shade_smooth(riser)
        parts.append(riser)
        neck_z = 0.014 + self.rise * 0.72
        neck = cylinder(
            self.pipe_r * 0.95,
            self.reach,
            location=(self.reach / 2, 0, neck_z),
            rotation=(0, np.pi / 2, 0),
            name="kmix_neck",
        )
        assign(neck, chrome)
        shade_smooth(neck)
        parts.append(neck)
        nozzle = cylinder(
            self.pipe_r * 1.15,
            0.028,
            location=(self.reach, 0, neck_z - 0.012),
            name="kmix_nozzle",
        )
        assign(nozzle, chrome)
        shade_smooth(nozzle)
        parts.append(nozzle)

        if self.style == "lever":
            stem = cylinder(
                self.pipe_r * 0.7,
                0.032,
                location=(0.008, 0.028, 0.055),
                rotation=(np.pi / 2, 0, 0),
                name="kmix_stem",
            )
            assign(stem, chrome)
            shade_smooth(stem)
            lever = box(
                (0.012, 0.055, 0.014),
                location=(0.008, 0.055, 0.055),
                name="kmix_lever",
            )
            assign(lever, chrome)
            butil.modify_mesh(lever, "BEVEL", width=0.002, segments=2)
            parts.extend([stem, lever])
        else:
            for sign in (-1, 1):
                knob = cylinder(
                    self.pipe_r * 1.5,
                    0.022,
                    location=(0.0, sign * 0.045, 0.028),
                    name="kmix_knob",
                )
                assign(knob, chrome)
                shade_smooth(knob)
                parts.append(knob)

        obj = join_objects(parts)
        obj.name = f"KitchenMixerFactory({self.factory_seed}).mixer"
        return obj
