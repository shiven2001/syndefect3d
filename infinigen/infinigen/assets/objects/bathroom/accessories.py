"""Small bathroom accessories: exhaust fan, floor drain, toilet-paper holder, medicine cabinet."""

import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.wall_decorations.primitives import (
    assign,
    box,
    cylinder,
    mirror_material,
    shade_smooth,
    solid_material,
)
from infinigen.assets.utils.object import join_objects, new_bbox
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class ExhaustFanFactory(AssetFactory):
    """Ceiling extractor: square louver grille or round axial face (~160–220 mm)."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.round = uniform() < 0.4
            self.size = uniform(0.16, 0.22)
            self.thick = uniform(0.018, 0.028)
            self.n_slats = int(uniform(6, 10))
            tone = uniform(0.90, 0.97)
            self.color = (tone, tone, tone * 0.98)

    def create_placeholder(self, **params):
        s = self.size / 2
        return new_bbox(-s, s, -s, s, -self.thick - 0.002, 0.002)

    def create_asset(self, **params):
        body = solid_material(
            f"ExhaustBody_{self.factory_seed}", self.color, roughness=0.36
        )
        dark = solid_material(
            f"ExhaustSlot_{self.factory_seed}", (0.10, 0.10, 0.10), roughness=0.55
        )
        s, t = self.size, self.thick
        parts = []
        if self.round:
            frame = cylinder(s / 2, t, location=(0, 0, -t / 2), name="exhaust_frame")
            assign(frame, body)
            shade_smooth(frame)
            parts.append(frame)
            well = cylinder(s * 0.38, 0.006, location=(0, 0, -t + 0.004), name="exhaust_well")
            assign(well, dark)
            parts.append(well)
            hub = cylinder(s * 0.10, 0.008, location=(0, 0, -t * 0.40), name="exhaust_hub")
            assign(hub, body)
            shade_smooth(hub)
            parts.append(hub)
            for i in range(self.n_slats):
                ang = i * (np.pi / self.n_slats)
                slat = box(
                    (s * 0.72, 0.004, 0.007),
                    location=(0, 0, -t * 0.42),
                    name=f"exhaust_slat_{i}",
                )
                slat.rotation_euler[2] = ang
                butil.apply_transform(slat, loc=True)
                assign(slat, body)
                parts.append(slat)
        else:
            frame = box((s, s, t), location=(0, 0, -t / 2), name="exhaust_frame")
            assign(frame, body)
            butil.modify_mesh(frame, "BEVEL", width=0.004, segments=2)
            parts.append(frame)
            inset = s * 0.78
            well = box((inset, inset, 0.006), location=(0, 0, -t + 0.004), name="exhaust_well")
            assign(well, dark)
            parts.append(well)
            slat_w = inset / (self.n_slats + 1)
            for i in range(self.n_slats):
                y = -inset / 2 + slat_w * (i + 1)
                slat = box(
                    (inset * 0.90, 0.004, 0.007),
                    location=(0, y, -t * 0.45),
                    name=f"exhaust_slat_{i}",
                )
                assign(slat, body)
                parts.append(slat)
        obj = join_objects(parts)
        obj.name = f"ExhaustFanFactory({self.factory_seed}).fan"
        return obj


class FloorDrainFactory(AssetFactory):
    """Stainless floor waste, 100–130 mm, square grid or round slots."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.round = uniform() < 0.35
            self.size = uniform(0.10, 0.13)
            self.thick = 0.008
            self.n = int(uniform(5, 8))

    def create_placeholder(self, **params):
        s = self.size / 2
        return new_bbox(-s, s, -s, s, 0, self.thick)

    def create_asset(self, **params):
        steel = solid_material(
            f"DrainSteel_{self.factory_seed}",
            (0.62, 0.63, 0.64),
            roughness=0.26,
            metallic=1.0,
        )
        dark = solid_material(
            f"DrainHole_{self.factory_seed}", (0.05, 0.05, 0.05), roughness=0.7
        )
        s, t = self.size, self.thick
        # on_floor uses a 1 cm margin; sink the grille so it sits ~2 mm proud.
        z = -0.008
        parts = []
        if self.round:
            frame = cylinder(s / 2, t, location=(0, 0, z), name="drain_frame")
            assign(frame, steel)
            shade_smooth(frame)
            parts.append(frame)
            well = cylinder(s * 0.38, t * 0.5, location=(0, 0, z - 0.002), name="drain_well")
            assign(well, dark)
            parts.append(well)
            for i in range(self.n):
                ang = i * (np.pi / self.n)
                bar = box(
                    (s * 0.72, 0.004, t * 0.7),
                    location=(0, 0, z + 0.001),
                    name="drain_bar",
                )
                bar.rotation_euler[2] = ang
                butil.apply_transform(bar, loc=True)
                assign(bar, steel)
                parts.append(bar)
        else:
            frame = box((s, s, t), location=(0, 0, z), name="drain_frame")
            assign(frame, steel)
            butil.modify_mesh(frame, "BEVEL", width=0.0015, segments=2)
            parts.append(frame)
            inner = s * 0.76
            well = box((inner, inner, t * 0.5), location=(0, 0, z - 0.002), name="drain_well")
            assign(well, dark)
            parts.append(well)
            bar_w = 0.004
            for i in range(self.n):
                y = -inner / 2 + inner * (i + 0.5) / self.n
                bar = box(
                    (inner * 0.90, bar_w, t * 0.7),
                    location=(0, y, z + 0.001),
                    name="drain_bar",
                )
                assign(bar, steel)
                parts.append(bar)
        obj = join_objects(parts)
        obj.name = f"FloorDrainFactory({self.factory_seed}).drain"
        return obj


class ToiletPaperHolderFactory(AssetFactory):
    """Chrome wall bar with a paper roll. Mount ~650 mm AFF."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.roll_r = uniform(0.052, 0.060)
            self.roll_w = uniform(0.095, 0.110)
            self.arm = uniform(0.070, 0.090)
            self.post_h = uniform(0.011, 0.015)

    def create_placeholder(self, **params):
        return new_bbox(
            -0.002,
            self.arm + self.roll_r + 0.02,
            -self.roll_w / 2 - 0.03,
            self.roll_w / 2 + 0.03,
            -self.roll_r - 0.02,
            self.roll_r + 0.03,
        )

    def create_asset(self, **params):
        chrome = solid_material(
            f"TPChrome_{self.factory_seed}",
            (0.74, 0.75, 0.77),
            roughness=0.10,
            metallic=1.0,
        )
        paper = solid_material(
            f"TPPaper_{self.factory_seed}",
            (0.94, 0.93, 0.89),
            roughness=0.78,
        )
        plate = box((0.006, 0.055, 0.040), location=(0.003, 0, 0), name="tp_plate")
        assign(plate, chrome)
        butil.modify_mesh(plate, "BEVEL", width=0.0015, segments=2)
        parts = [plate]
        for sign in (-1, 1):
            arm = cylinder(
                self.post_h,
                self.arm,
                location=(self.arm / 2, sign * (self.roll_w / 2 + 0.008), 0),
                rotation=(0, np.pi / 2, 0),
                name="tp_arm",
            )
            assign(arm, chrome)
            shade_smooth(arm)
            parts.append(arm)
            cap = cylinder(
                self.post_h * 1.15,
                0.008,
                location=(self.arm - 0.002, sign * (self.roll_w / 2 + 0.008), 0),
                rotation=(0, np.pi / 2, 0),
                name="tp_cap",
            )
            assign(cap, chrome)
            shade_smooth(cap)
            parts.append(cap)
        bar = cylinder(
            self.post_h * 0.65,
            self.roll_w + 0.010,
            location=(self.arm * 0.92, 0, 0),
            rotation=(np.pi / 2, 0, 0),
            name="tp_bar",
        )
        assign(bar, chrome)
        shade_smooth(bar)
        parts.append(bar)
        roll = cylinder(
            self.roll_r,
            self.roll_w,
            location=(self.arm * 0.92, 0, -0.006),
            rotation=(np.pi / 2, 0, 0),
            name="tp_roll",
        )
        assign(roll, paper)
        shade_smooth(roll)
        parts.append(roll)
        core = cylinder(
            0.019,
            self.roll_w + 0.004,
            location=(self.arm * 0.92, 0, -0.006),
            rotation=(np.pi / 2, 0, 0),
            name="tp_core",
        )
        assign(
            core,
            solid_material(
                f"TPCore_{self.factory_seed}", (0.72, 0.52, 0.26), roughness=0.62
            ),
        )
        parts.append(core)
        obj = join_objects(parts)
        obj.name = f"ToiletPaperHolderFactory({self.factory_seed}).holder"
        return obj


class MedicineCabinetFactory(AssetFactory):
    """Mirrored wall cabinet, ~360–500 × 520–680 × 100–140 mm, typical above-sink."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.width = uniform(0.36, 0.50)
            self.height = uniform(0.52, 0.68)
            self.depth = uniform(0.10, 0.14)
            tone = uniform(0.90, 0.97)
            self.body = (tone, tone, tone * 0.98)

    def create_placeholder(self, **params):
        return new_bbox(
            -0.002,
            self.depth,
            -self.width / 2,
            self.width / 2,
            -self.height / 2,
            self.height / 2,
        )

    def create_asset(self, **params):
        body_mat = solid_material(
            f"MedCabBody_{self.factory_seed}", self.body, roughness=0.30
        )
        mirror = mirror_material(f"MedCabMirror_{self.factory_seed}")
        chrome = solid_material(
            f"MedCabChrome_{self.factory_seed}",
            (0.74, 0.75, 0.77),
            roughness=0.12,
            metallic=1.0,
        )
        w, h, d = self.width, self.height, self.depth
        carcass = box((d * 0.92, w, h), location=(d * 0.46, 0, 0), name="medcab_body")
        assign(carcass, body_mat)
        butil.modify_mesh(carcass, "BEVEL", width=0.003, segments=2)
        parts = [carcass]
        door = box(
            (0.005, w - 0.016, h - 0.016),
            location=(d - 0.004, 0, 0),
            name="medcab_door",
        )
        assign(door, mirror)
        parts.append(door)
        frame_t = 0.012
        for size, loc in (
            ((0.008, w, frame_t), (d - 0.002, 0, h / 2 - frame_t / 2)),
            ((0.008, w, frame_t), (d - 0.002, 0, -h / 2 + frame_t / 2)),
            ((0.008, frame_t, h), (d - 0.002, w / 2 - frame_t / 2, 0)),
            ((0.008, frame_t, h), (d - 0.002, -w / 2 + frame_t / 2, 0)),
        ):
            f = box(size, loc, name="medcab_frame")
            assign(f, body_mat)
            parts.append(f)
        handle = cylinder(
            0.005,
            0.10,
            location=(d + 0.008, w * 0.34, 0),
            name="medcab_handle",
        )
        assign(handle, chrome)
        shade_smooth(handle)
        parts.append(handle)
        obj = join_objects(parts)
        obj.name = f"MedicineCabinetFactory({self.factory_seed}).cabinet"
        return obj
