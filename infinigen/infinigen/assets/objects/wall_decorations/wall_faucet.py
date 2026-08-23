# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.

"""Procedural wall-mounted bathroom mixer. Materials match TapFactory metals."""

import numpy as np
from numpy.random import uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.table_decorations.sink import TapFactory
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
from infinigen.core.util.random import weighted_sample


class WallFaucetFactory(AssetFactory):
    """Wall mixer: backplate + spout + one or two handles.

    Sink-mounted taps still come from ``TapFactory``. This factory is the
    wall-mounted counterpart used as bathroom wall decoration.
    """

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.style = "lever" if uniform() < 0.55 else "two_handle"
            self.plate_w = uniform(0.16, 0.22) if self.style == "two_handle" else uniform(0.10, 0.14)
            self.plate_h = uniform(0.07, 0.10)
            self.plate_t = uniform(0.008, 0.014)
            self.reach = uniform(0.12, 0.18)
            self.drop = uniform(0.04, 0.08)
            self.pipe_r = uniform(0.009, 0.013)
            self.handle_len = uniform(0.045, 0.07)
            # Same metal sampling as the sink-mounted TapFactory.
            self.tap_factory = TapFactory(factory_seed)
            self.metal = self.tap_factory.params.get("Tap")
            if self.metal is None:
                self.metal = weighted_sample(material_assignments.metals)()()

    def create_placeholder(self, **params):
        return new_bbox(
            -0.002,
            self.reach + 0.02,
            -self.plate_w / 2 - 0.02,
            self.plate_w / 2 + 0.02,
            -self.plate_h / 2 - self.drop,
            self.plate_h / 2 + 0.03,
        )

    def create_asset(self, **params):
        chrome = self.metal
        if chrome is None:
            chrome = solid_material(
                f"WallFaucetMetal_{self.factory_seed}",
                (0.72, 0.73, 0.75),
                roughness=0.12,
                metallic=1.0,
            )

        plate = box(
            (self.plate_t, self.plate_w, self.plate_h),
            location=(self.plate_t / 2, 0, 0),
            name="wallfaucet_plate",
        )
        assign(plate, chrome)
        butil.modify_mesh(plate, "BEVEL", width=0.002, segments=2)
        shade_smooth(plate)
        parts = [plate]

        # Horizontal spout, then a downward nozzle.
        spout = cylinder(
            self.pipe_r,
            self.reach * 0.72,
            location=(self.plate_t + self.reach * 0.36, 0, -0.008),
            rotation=(0, np.pi / 2, 0),
            name="wallfaucet_spout",
        )
        assign(spout, chrome)
        shade_smooth(spout)
        parts.append(spout)

        elbow = cylinder(
            self.pipe_r * 1.05,
            self.pipe_r * 2.2,
            location=(self.plate_t + self.reach * 0.72, 0, -0.008),
            rotation=(0, np.pi / 2, 0),
            name="wallfaucet_elbow",
        )
        assign(elbow, chrome)
        parts.append(elbow)

        nozzle = cylinder(
            self.pipe_r * 0.85,
            self.drop,
            location=(
                self.plate_t + self.reach * 0.72,
                0,
                -0.008 - self.drop / 2,
            ),
            name="wallfaucet_nozzle",
        )
        assign(nozzle, chrome)
        shade_smooth(nozzle)
        parts.append(nozzle)

        aerator = cylinder(
            self.pipe_r * 1.05,
            0.008,
            location=(
                self.plate_t + self.reach * 0.72,
                0,
                -0.008 - self.drop,
            ),
            name="wallfaucet_aerator",
        )
        assign(aerator, chrome)
        parts.append(aerator)

        if self.style == "lever":
            stem = cylinder(
                self.pipe_r * 0.7,
                0.028,
                location=(self.plate_t + 0.02, 0, self.plate_h * 0.22),
                rotation=(0, np.pi / 2, 0),
                name="wallfaucet_stem",
            )
            assign(stem, chrome)
            lever = box(
                (0.012, 0.014, self.handle_len),
                location=(
                    self.plate_t + 0.034,
                    0,
                    self.plate_h * 0.22 + self.handle_len / 2 - 0.008,
                ),
                name="wallfaucet_lever",
            )
            assign(lever, chrome)
            butil.modify_mesh(lever, "BEVEL", width=0.002, segments=2)
            parts.extend([stem, lever])
        else:
            for sign in (-1, 1):
                knob = cylinder(
                    self.pipe_r * 1.6,
                    0.028,
                    location=(
                        self.plate_t + 0.02,
                        sign * (self.plate_w * 0.32),
                        0.0,
                    ),
                    rotation=(0, np.pi / 2, 0),
                    name="wallfaucet_knob",
                )
                assign(knob, chrome)
                shade_smooth(knob)
                bar = box(
                    (0.008, 0.008, self.handle_len * 0.7),
                    location=(
                        self.plate_t + 0.036,
                        sign * (self.plate_w * 0.32),
                        self.handle_len * 0.15,
                    ),
                    name="wallfaucet_bar",
                )
                assign(bar, chrome)
                parts.extend([knob, bar])

        obj = join_objects(parts)
        obj.name = f"WallFaucetFactory({self.factory_seed}).tap"
        return obj
