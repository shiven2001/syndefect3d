"""Glass-ceramic / induction cooktop for handover kitchen counters."""

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


class CooktopFactory(AssetFactory):
    """Flush-mount hob: dark glass plate, burner rings, front knobs."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.n_burners = 2 if uniform() < 0.45 else 4
            self.width = uniform(0.52, 0.62) if self.n_burners == 2 else uniform(0.56, 0.72)
            self.depth = uniform(0.48, 0.54)
            self.thick = uniform(0.038, 0.052)
            self.frame = uniform(0.012, 0.018)

    def create_placeholder(self, **params):
        return new_bbox(
            -self.depth / 2,
            self.depth / 2,
            -self.width / 2,
            self.width / 2,
            0,
            self.thick,
        )

    def create_asset(self, **params):
        glass = solid_material(
            f"HobGlass_{self.factory_seed}",
            (0.04, 0.04, 0.045),
            roughness=0.12,
            metallic=0.15,
        )
        steel = solid_material(
            f"HobSteel_{self.factory_seed}",
            (0.62, 0.63, 0.64),
            roughness=0.22,
            metallic=1.0,
        )
        ring_mat = solid_material(
            f"HobRing_{self.factory_seed}",
            (0.18, 0.18, 0.19),
            roughness=0.35,
            metallic=0.4,
        )
        w, d, t = self.width, self.depth, self.thick
        plate = box((d, w, t * 0.55), location=(0, 0, t * 0.55 / 2), name="hob_glass")
        assign(plate, glass)
        butil.modify_mesh(plate, "BEVEL", width=0.003, segments=2)
        parts = [plate]
        trim = box(
            (d + self.frame, w + self.frame, t * 0.22),
            location=(0, 0, t * 0.08),
            name="hob_frame",
        )
        assign(trim, steel)
        parts.append(trim)

        if self.n_burners == 2:
            layout = ((-d * 0.12, -w * 0.18, 0.09), (-d * 0.12, w * 0.18, 0.075))
        else:
            layout = (
                (-d * 0.14, -w * 0.22, 0.085),
                (-d * 0.14, w * 0.22, 0.070),
                (d * 0.10, -w * 0.20, 0.062),
                (d * 0.10, w * 0.20, 0.078),
            )
        z_ring = t * 0.58
        for i, (x, y, r) in enumerate(layout):
            ring = cylinder(r, 0.003, location=(x, y, z_ring), name=f"hob_ring_{i}")
            assign(ring, ring_mat)
            shade_smooth(ring)
            parts.append(ring)
            inner = cylinder(r * 0.45, 0.002, location=(x, y, z_ring + 0.001), name=f"hob_dot_{i}")
            assign(inner, steel)
            parts.append(inner)

        n_knob = self.n_burners
        span = w * 0.55
        for i in range(n_knob):
            y = -span / 2 + span * i / max(n_knob - 1, 1) if n_knob > 1 else 0
            knob = cylinder(
                0.011,
                0.014,
                location=(d * 0.38, y, t * 0.62),
                name=f"hob_knob_{i}",
            )
            assign(knob, steel)
            shade_smooth(knob)
            parts.append(knob)

        obj = join_objects(parts)
        obj.name = f"CooktopFactory({self.factory_seed}).hob"
        return obj
