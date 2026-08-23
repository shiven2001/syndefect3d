# Apartment structural column (painted plaster / concrete), not a classical order.

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.object import join_objects, new_cube
from infinigen.core import surface
from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample


class PillarFactory(AssetFactory):
    """Floor-to-ceiling square/rect column, same language as painted apartment walls."""

    def __init__(self, factory_seed, coarse=False, constants=None):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            if constants is None:
                constants = RoomConstants()
            self.height = constants.wall_height - constants.wall_thickness
            wall_t = max(float(constants.wall_thickness), 0.18)
            self.width = float(np.clip(wall_t * uniform(1.35, 1.85), 0.22, 0.38))
            self.depth = float(np.clip(self.width * uniform(0.95, 1.05), 0.22, 0.38))
            self.skirt_h = uniform(0.08, 0.12)
            self.edge_bevel = uniform(0.004, 0.010)
            self.surface = weighted_sample(material_assignments.wall_plaster)()

    def create_asset(self, **params) -> bpy.types.Object:
        shaft = new_cube()
        shaft.scale = (self.width / 2, self.depth / 2, self.height / 2)
        shaft.location = (0.0, 0.0, self.height / 2)
        butil.apply_transform(shaft, loc=True)
        butil.modify_mesh(shaft, "BEVEL", width=self.edge_bevel, segments=2)
        parts = [shaft]

        extra = uniform(0.006, 0.012)
        skirt = new_cube()
        skirt.scale = (
            (self.width + extra) / 2,
            (self.depth + extra) / 2,
            self.skirt_h / 2,
        )
        skirt.location = (0.0, 0.0, self.skirt_h / 2)
        butil.apply_transform(skirt, loc=True)
        parts.append(skirt)

        obj = join_objects(parts)
        obj.name = f"ApartmentPillar_{self.factory_seed}"
        surface.assign_material(obj, self.surface())
        return obj

    def finalize_assets(self, assets):
        surface.assign_material(assets, self.surface())
