"""UK / HK rocker light switch on a plastic faceplate."""

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


class WallSwitchFactory(AssetFactory):
    """1–3 gang rocker switch, ~86 mm module, typical 1150–1250 mm AFF."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.n_gangs = int(uniform(1, 3.6))
            # A 1-3 gang plate is a single ~86 mm square: extra gangs divide that
            # face into narrower rockers, they do not widen the plate. Scaling the
            # plate per gang gave a 180 mm slab for a 2-gang switch.
            self.module = uniform(0.082, 0.088)
            self.gap = uniform(0.0025, 0.004)
            self.plate_h = self.module
            self.plate_w = self.module
            self.gang_pitch = self.module * 0.76 / self.n_gangs
            self.thickness = uniform(0.008, 0.012)
            tone = uniform(0.86, 0.97)
            self.plate_color = (tone, tone * uniform(0.98, 1.0), tone * uniform(0.94, 1.0))
            rocker_v = tone * uniform(0.94, 1.02)
            self.rocker_color = (rocker_v, rocker_v, rocker_v * 0.98)

    def create_placeholder(self, **params):
        return new_bbox(
            -0.002,
            self.thickness + 0.004,
            -self.plate_w / 2,
            self.plate_w / 2,
            -self.plate_h / 2,
            self.plate_h / 2,
        )

    def create_asset(self, **params):
        plate_mat = plastic_material(
            f"WallSwitchPlate_{self.factory_seed}",
            self.plate_color,
            roughness=uniform(0.30, 0.42),
        )
        rocker_mat = plastic_material(
            f"WallSwitchRocker_{self.factory_seed}",
            self.rocker_color,
            roughness=0.34,
        )
        screw_mat = solid_material(
            f"WallSwitchScrew_{self.factory_seed}",
            (0.55, 0.55, 0.57),
            roughness=0.42,
            metallic=1.0,
        )
        # Faceplate: generously rounded corners, as moulded plates have.
        plate = rounded_box(
            (self.thickness, self.plate_w, self.plate_h),
            location=(self.thickness / 2, 0, 0),
            radius=min(0.006, self.thickness * 0.45),
            segments=4,
            name="wallswitch_plate",
        )
        assign(plate, plate_mat)
        parts = [plate]
        span = self.gang_pitch
        rocker_w = max(0.008, span - self.gap)
        for i in range(self.n_gangs):
            y = (i - (self.n_gangs - 1) / 2) * span
            # Rockers sit in a shallow well and bulge outward, so the paddle
            # catches light along a curve instead of reading as a flat tile.
            well = rounded_box(
                (0.0015, rocker_w + 0.0035, self.module * 0.62),
                location=(self.thickness - 0.0005, y, 0.0),
                radius=0.0015,
                segments=2,
                name=f"wallswitch_well_{i}",
            )
            assign(well, plate_mat)
            parts.append(well)

            rocker = rounded_box(
                (0.006, rocker_w, self.module * 0.55),
                location=(self.thickness + 0.0025, y, 0.0),
                radius=0.0028,
                segments=4,
                name=f"wallswitch_rocker_{i}",
            )
            # Slight tilt: a rocker is never perfectly flush with the plate.
            rocker.rotation_euler = (0, uniform(-0.05, 0.05), 0)
            butil.apply_transform(rocker)
            assign(rocker, rocker_mat)
            shade_smooth(rocker)
            parts.append(rocker)
        for sign in (-1, 1):
            # Countersunk fixing screws, sunk just below the plate face.
            screw = rounded_box(
                (0.0012, 0.0055, 0.0055),
                location=(
                    self.thickness - 0.0004,
                    sign * (self.plate_w / 2 - 0.007),
                    0.0,
                ),
                radius=0.0008,
                segments=3,
                name="wallswitch_screw",
            )
            assign(screw, screw_mat)
            parts.append(screw)
        obj = join_objects(parts)
        obj.name = f"WallSwitchFactory({self.factory_seed}).switch"
        return obj
