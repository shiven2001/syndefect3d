# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.

"""Procedural split-system indoor AC head, optional side pipe trunk."""

import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.wall_decorations.primitives import (
    assign,
    box,
    plastic_material,
    rounded_box,
    shade_smooth,
    solid_material,
)
from infinigen.assets.utils.object import join_objects, new_bbox
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class SplitACFactory(AssetFactory):
    """High-wall indoor unit: housing, intake slats, outlet vanes, optional trunk."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.width = uniform(1.05, 1.32)
            self.height = uniform(0.40, 0.50)
            self.depth = uniform(0.22, 0.28)
            self.n_intake = int(uniform(8, 13))
            self.n_outlet = int(uniform(4, 7))
            self.has_trunk = uniform() < 0.75
            self.trunk_side = 1.0 if uniform() < 0.5 else -1.0
            # Keep the side trunk at the previous scale; only the head grows.
            self.trunk_len = uniform(0.18, 0.40)
            self.trunk_h = uniform(0.12, 0.20)
            self.trunk_d = uniform(0.055, 0.075)
            self.trunk_rise = uniform(0.05, 0.09)
            tone = uniform(0.88, 0.98)
            self.body_color = (tone, tone, tone * uniform(0.98, 1.0))
            self.grille_color = (
                tone * uniform(0.78, 0.90),
                tone * uniform(0.78, 0.90),
                tone * uniform(0.78, 0.92),
            )
            self.trunk_color = (
                tone * uniform(0.90, 1.0),
                tone * uniform(0.90, 1.0),
                tone * uniform(0.88, 0.98),
            )
            led_on = uniform() < 0.6
            self.led_color = (0.15, 0.85, 0.35) if led_on else (0.12, 0.12, 0.12)

    def _y_extent(self):
        extra = self.trunk_len if self.has_trunk else 0.0
        lo = -self.width / 2
        hi = self.width / 2
        if self.has_trunk:
            if self.trunk_side > 0:
                hi += extra
            else:
                lo -= extra
        return lo, hi

    def create_placeholder(self, **params):
        y0, y1 = self._y_extent()
        z_top = self.height / 2
        if self.has_trunk:
            z_top = max(
                z_top,
                -self.height * 0.08 + self.trunk_h / 2 + self.trunk_rise,
            )
        return new_bbox(
            -0.002,
            self.depth,
            y0,
            y1,
            -self.height / 2,
            z_top,
        )

    def create_asset(self, **params):
        body_mat = plastic_material(
            f"SplitACBody_{self.factory_seed}",
            self.body_color,
            roughness=uniform(0.26, 0.36),
            sheen_scale=140.0,
        )
        grille_mat = plastic_material(
            f"SplitACGrille_{self.factory_seed}",
            self.grille_color,
            roughness=0.48,
        )
        trunk_mat = plastic_material(
            f"SplitACTrunk_{self.factory_seed}",
            self.trunk_color,
            roughness=0.42,
        )
        led_mat = solid_material(
            f"SplitACLed_{self.factory_seed}",
            self.led_color,
            roughness=0.15,
        )

        housing = rounded_box(
            (self.depth * 0.92, self.width, self.height),
            location=(self.depth * 0.46, 0, 0),
            radius=min(self.height, self.depth) * 0.30,
            segments=8,
            name="splitac_housing",
        )
        assign(housing, body_mat)
        parts = [housing]

        # Upper intake slats (front face).
        intake_h = self.height * 0.42
        intake_z0 = self.height * 0.08
        slat_t = 0.003
        slat_gap = intake_h / max(self.n_intake, 1)
        for i in range(self.n_intake):
            z = intake_z0 + (i + 0.5) * slat_gap
            slat = box(
                (0.004, self.width * 0.86, slat_t),
                location=(self.depth * 0.93, 0, z),
                name=f"splitac_intake_{i}",
            )
            assign(slat, grille_mat)
            parts.append(slat)

        # Lower outlet vanes, slightly angled.
        outlet_h = self.height * 0.28
        outlet_z0 = -self.height * 0.42
        vane_gap = outlet_h / max(self.n_outlet, 1)
        tilt = uniform(0.18, 0.38)
        for i in range(self.n_outlet):
            z = outlet_z0 + (i + 0.5) * vane_gap
            vane = box(
                (0.018, self.width * 0.82, 0.006),
                location=(self.depth * 0.88, 0, z),
                name=f"splitac_vane_{i}",
            )
            vane.rotation_euler = (0, tilt, 0)
            butil.apply_transform(vane)
            assign(vane, grille_mat)
            parts.append(vane)

        divider = rounded_box(
            (0.008, self.width * 0.88, 0.006),
            location=(self.depth * 0.94, 0, -self.height * 0.04),
            radius=0.002,
            segments=3,
            name="splitac_divider",
        )
        assign(divider, grille_mat)
        parts.append(divider)

        # Display window sits flush in the lower fascia, right of centre.
        led = rounded_box(
            (0.003, 0.028, 0.006),
            location=(self.depth * 0.95, self.width * 0.38, -self.height * 0.08),
            radius=0.001,
            segments=2,
            name="splitac_led",
        )
        assign(led, led_mat)
        parts.append(led)

        if self.has_trunk:
            y_face = self.trunk_side * (self.width / 2)
            y_c = y_face + self.trunk_side * (self.trunk_len / 2)
            trunk = box(
                (self.trunk_d, self.trunk_len, self.trunk_h),
                location=(self.trunk_d / 2, y_c, -self.height * 0.08),
                name="splitac_trunk",
            )
            assign(trunk, trunk_mat)
            butil.modify_mesh(trunk, "BEVEL", width=0.004, segments=2)
            parts.append(trunk)
            # Short vertical riser so it does not look like a floating stub.
            rise = self.trunk_rise
            riser = box(
                (self.trunk_d * 0.92, self.trunk_d * 1.1, rise),
                location=(
                    self.trunk_d / 2,
                    y_face + self.trunk_side * (self.trunk_len - self.trunk_d * 0.55),
                    -self.height * 0.08 + self.trunk_h / 2 + rise / 2,
                ),
                name="splitac_riser",
            )
            assign(riser, trunk_mat)
            parts.append(riser)

        obj = join_objects(parts)
        obj.name = f"SplitACFactory({self.factory_seed}).ac"
        return obj
