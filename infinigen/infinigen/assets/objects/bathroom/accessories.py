"""Small bathroom accessories: exhaust fan, floor drain, toilet-paper holder, medicine cabinet."""

import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.wall_decorations.primitives import (
    assign,
    box,
    cylinder,
    mirror_material,
    plastic_material,
    shade_smooth,
    solid_material,
)
from infinigen.assets.utils.object import join_objects, new_bbox
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class ExhaustFanFactory(AssetFactory):
    """Ceiling extractor: square louver grille or round axial face (~240–300 mm)."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            # Flats fit a square louvred grille; the round axial face is rare.
            self.round = uniform() < 0.15
            self.size = uniform(0.24, 0.30)
            # Sits nearly flush, like a ceiling light.
            self.thick = uniform(0.010, 0.014)
            self.n_slats = int(uniform(12, 17))
            # Slight set so the slots read, without raking the blades into the ceiling.
            self.tilt = uniform(0.18, 0.26)
            tone = uniform(0.90, 0.96)
            self.color = (tone, tone, tone * 0.985)

    def create_placeholder(self, **params):
        s = self.size / 2 + 0.010
        return new_bbox(-s, s, -s, s, -self.thick - 0.004, 0.002)

    def _louver(self, length, width, thick, y, z, mat, name="exhaust_slat"):
        """Tilt a slat about its own centre, then move it into the grille."""
        slat = box((length, width, thick), location=(0.0, 0.0, 0.0), name=name)
        slat.rotation_euler = (self.tilt, 0.0, 0.0)
        butil.apply_transform(slat)
        slat.location = (0.0, y, z)
        butil.apply_transform(slat, loc=True)
        assign(slat, mat)
        return slat

    def create_asset(self, **params):
        body = plastic_material(
            f"ExhaustBody_{self.factory_seed}", self.color, roughness=0.36
        )
        dark = solid_material(
            f"ExhaustSlot_{self.factory_seed}", (0.12, 0.12, 0.13), roughness=0.72
        )
        s, t = self.size, self.thick
        parts = []
        flange = s * 0.08
        outer = s / 2 + 0.010
        opening = 2 * (outer - flange)
        # Overlapping louvers: the dark well only shows as thin slots.
        n = self.n_slats
        pitch = opening / n
        slat_w = pitch * 1.08
        slat_z = -t * 0.40

        if self.round:
            well = cylinder(
                outer - flange * 0.15,
                0.003,
                location=(0, 0, -0.003),
                name="exhaust_plenum",
            )
            assign(well, dark)
            parts.append(well)
            ring = cylinder(outer, t, location=(0, 0, -t / 2), name="exhaust_flange")
            hole = cylinder(
                outer - flange, t * 3, location=(0, 0, -t / 2), name="exhaust_cut"
            )
            butil.modify_mesh(ring, "BOOLEAN", object=hole, operation="DIFFERENCE")
            butil.delete(hole)
            assign(ring, body)
            shade_smooth(ring)
            parts.append(ring)
            R = outer - flange
            for i in range(n):
                y = -opening / 2 + pitch * (i + 0.5)
                if abs(y) >= R * 0.98:
                    continue
                half = 2.0 * float(np.sqrt(max(R * R - y * y, 1e-8))) * 0.96
                parts.append(self._louver(half, slat_w, 0.0024, y, slat_z, body))
            hub = cylinder(R * 0.12, 0.004, location=(0, 0, slat_z), name="exhaust_hub")
            assign(hub, body)
            shade_smooth(hub)
            parts.append(hub)
        else:
            well = box(
                (opening * 0.98, opening * 0.98, 0.003),
                location=(0, 0, -0.003),
                name="exhaust_plenum",
            )
            assign(well, dark)
            parts.append(well)
            for dx, dy, sx, sy in (
                (0, outer - flange / 2, 2 * outer, flange),
                (0, -(outer - flange / 2), 2 * outer, flange),
                (outer - flange / 2, 0, flange, 2 * outer - 2 * flange),
                (-(outer - flange / 2), 0, flange, 2 * outer - 2 * flange),
            ):
                piece = box((sx, sy, t), location=(dx, dy, -t / 2), name="exhaust_flange")
                butil.modify_mesh(piece, "BEVEL", width=0.0015, segments=2)
                assign(piece, body)
                parts.append(piece)
            for i in range(n):
                y = -opening / 2 + pitch * (i + 0.5)
                parts.append(
                    self._louver(opening * 0.98, slat_w, 0.0024, y, slat_z, body)
                )

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
            # A floor gully is set into the screed. The body used to be ~50 mm
            # deep while the placeholder claimed 8 mm, so the solver stood the
            # whole thing on the floor instead of letting it sit flush.
            self.thick = 0.006
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
            f"DrainHole_{self.factory_seed}", (0.03, 0.03, 0.035), roughness=0.85
        )
        s, t = self.size, self.thick
        # Everything lives in a thin slab just above the floor plane: the floor
        # is a plane with no thickness, so anything modelled below it would be
        # hidden anyway, and a deep body just lifts the grate off the tiles.
        z = t / 2
        parts = []

        # Shallow dark pan right under the bars, which is what gives the grate
        # its openings - a solid rim over a buried sump read as a steel tile.
        sump_r = s * 0.40
        if self.round:
            sump = cylinder(sump_r, 0.002, location=(0, 0, 0.001), name="drain_sump")
        else:
            sump = box((s * 0.80, s * 0.80, 0.002), location=(0, 0, 0.001), name="drain_sump")
        assign(sump, dark)
        parts.append(sump)

        rim_w = s * 0.10
        if self.round:
            outer = cylinder(s / 2, t, location=(0, 0, z), name="drain_rim")
            inner = cylinder(s / 2 - rim_w, t * 3, location=(0, 0, z), name="drain_rim_cut")
            butil.modify_mesh(outer, "BOOLEAN", object=inner, operation="DIFFERENCE")
            butil.delete(inner)
            assign(outer, steel)
            shade_smooth(outer)
            parts.append(outer)
            span = s - 2 * rim_w
            for i in range(self.n):
                bar = box(
                    (span, 0.0045, t * 0.8),
                    location=(0, 0, z + 0.0005),
                    name="drain_bar",
                )
                bar.rotation_euler[2] = i * (np.pi / self.n)
                butil.apply_transform(bar, loc=True)
                assign(bar, steel)
                parts.append(bar)
        else:
            # Four border pieces leave the sump visible between the slats.
            for dx, dy, sx, sy in (
                (0, (s - rim_w) / 2, s, rim_w),
                (0, -(s - rim_w) / 2, s, rim_w),
                ((s - rim_w) / 2, 0, rim_w, s - 2 * rim_w),
                (-(s - rim_w) / 2, 0, rim_w, s - 2 * rim_w),
            ):
                piece = box((sx, sy, t), location=(dx, dy, z), name="drain_rim")
                butil.modify_mesh(piece, "BEVEL", width=0.0012, segments=2)
                assign(piece, steel)
                parts.append(piece)
            span = s - 2 * rim_w
            pitch = span / self.n
            for i in range(self.n - 1):
                y = -span / 2 + pitch * (i + 1)
                bar = box(
                    (span, 0.0045, t * 0.8),
                    location=(0, y, z + 0.0005),
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
