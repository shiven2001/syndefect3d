"""Floor vanity cabinet: white laminate box, doors, drawer, chrome bar pulls.

Typical handover vanity ~580–780 × 400–480 × 800–860 mm.
"""

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


class VanityCabinetFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.width = uniform(0.58, 0.78)
            self.depth = uniform(0.42, 0.48)
            self.height = uniform(0.80, 0.86)
            self.top_t = uniform(0.022, 0.030)
            self.door_gap = 0.0025
            self.n_doors = 2
            tone = uniform(0.91, 0.97)
            self.body = (tone, tone * 0.995, tone * 0.98)

    def create_placeholder(self, **params):
        return new_bbox(-self.depth, 0, -self.width / 2, self.width / 2, 0, self.height)

    def create_asset(self, **params):
        body_mat = solid_material(
            f"VanityBody_{self.factory_seed}", self.body, roughness=0.30
        )
        top_mat = solid_material(
            f"VanityTop_{self.factory_seed}",
            (0.90, 0.91, 0.92),
            roughness=0.16,
        )
        chrome = solid_material(
            f"VanityChrome_{self.factory_seed}",
            (0.74, 0.75, 0.77),
            roughness=0.12,
            metallic=1.0,
        )
        w, d, h = self.width, self.depth, self.height
        kick = 0.08
        carcass_h = h - self.top_t - kick
        carcass = box(
            (d, w, carcass_h),
            location=(-d / 2, 0, kick + carcass_h / 2),
            name="vanity_body",
        )
        assign(carcass, body_mat)
        butil.modify_mesh(carcass, "BEVEL", width=0.002, segments=2)
        parts = [carcass]

        top = box(
            (d + 0.014, w + 0.018, self.top_t),
            location=(-d / 2 + 0.004, 0, h - self.top_t / 2),
            name="vanity_top",
        )
        assign(top, top_mat)
        butil.modify_mesh(top, "BEVEL", width=0.002, segments=2)
        parts.append(top)

        drawer_h = 0.12
        drawer = box(
            (0.016, w - 0.010, drawer_h),
            location=(-0.010, 0, h - self.top_t - 0.008 - drawer_h / 2),
            name="vanity_drawer",
        )
        assign(drawer, body_mat)
        butil.modify_mesh(drawer, "BEVEL", width=0.0015, segments=2)
        parts.append(drawer)
        d_pull = cylinder(
            0.005,
            0.14,
            location=(0.006, 0, h - self.top_t - 0.008 - drawer_h / 2),
            rotation=(1.5708, 0, 0),
            name="vanity_drawer_pull",
        )
        assign(d_pull, chrome)
        shade_smooth(d_pull)
        parts.append(d_pull)

        door_w = (w - self.door_gap * (self.n_doors + 1)) / self.n_doors
        door_h = carcass_h - drawer_h - 0.018
        door_z = kick + 0.006 + door_h / 2
        for i in range(self.n_doors):
            y = -w / 2 + self.door_gap + door_w / 2 + i * (door_w + self.door_gap)
            door = box(
                (0.016, door_w, door_h),
                location=(-0.010, y, door_z),
                name=f"vanity_door_{i}",
            )
            assign(door, body_mat)
            butil.modify_mesh(door, "BEVEL", width=0.0015, segments=2)
            parts.append(door)
            pull = cylinder(
                0.005,
                0.12,
                location=(0.006, y + (0.06 if i == 0 else -0.06), door_z + door_h * 0.08),
                name=f"vanity_pull_{i}",
            )
            assign(pull, chrome)
            shade_smooth(pull)
            parts.append(pull)

        plinth = box(
            (d - 0.05, w - 0.05, kick - 0.006),
            location=(-d / 2 + 0.012, 0, (kick - 0.006) / 2),
            name="vanity_plinth",
        )
        assign(plinth, body_mat)
        parts.append(plinth)

        obj = join_objects(parts)
        obj.name = f"VanityCabinetFactory({self.factory_seed}).vanity"
        return obj
