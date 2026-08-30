"""Corner walk-in shower: acrylic tray, chrome frame, L-shaped glass, rain head.

Typical small-apartment cubicle 800–900 mm, glass 1850–1950 mm.
Mixer ~1050–1150 mm AFF, rain head ~2000–2100 mm AFF, walk-in ~500–600 mm.
Placeholder matches BathtubFactory: back wall at min-X.
"""

import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.wall_decorations.primitives import (
    assign,
    box,
    cylinder,
    glass_material,
    shade_smooth,
    solid_material,
)
from infinigen.assets.utils.object import join_objects, new_bbox
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.assets.objects.bathroom.fittings import bathroom_chrome
from infinigen.core.util.math import FixedSeed


class ShowerStallFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.size = uniform(0.80, 0.92)  # into room (+X from back wall)
            self.width = uniform(0.82, 0.96)  # along wall (+Y)
            self.glass_h = uniform(1.85, 1.96)
            self.tray_h = uniform(0.050, 0.070)
            self.rim = uniform(0.032, 0.042)
            self.glass_t = uniform(0.006, 0.008)
            self.post = uniform(0.018, 0.024)
            self.opening = uniform(0.50, 0.60)
            self.head_r = uniform(0.105, 0.130)
            self.arm_len = uniform(0.30, 0.40)
            self.head_z = uniform(2.00, 2.10)
            self.mixer_z = uniform(1.05, 1.15)

    def create_placeholder(self, **params):
        return new_bbox(-self.size, 0, 0, self.width, 0, max(self.glass_h, self.head_z))

    def create_asset(self, **params):
        chrome = bathroom_chrome()
        acrylic = solid_material(
            f"ShowerTray_{self.factory_seed}",
            (0.93, 0.935, 0.94),
            roughness=0.22,
        )
        basin = solid_material(
            f"ShowerBasin_{self.factory_seed}",
            (0.86, 0.87, 0.88),
            roughness=0.28,
        )
        glass = glass_material(f"ShowerGlass_{self.factory_seed}")
        face_mat = solid_material(
            f"ShowerFace_{self.factory_seed}",
            (0.50, 0.51, 0.53),
            roughness=0.38,
            metallic=0.65,
        )
        parts = []

        d, w, th = self.size, self.width, self.tray_h
        x_wall = -d
        x_front = 0.0

        tray = box((d, w, th), location=(-d / 2, w / 2, th / 2), name="shower_tray")
        assign(tray, acrylic)
        butil.modify_mesh(tray, "BEVEL", width=0.006, segments=2)
        parts.append(tray)

        inner_d = d - self.rim * 2
        inner_w = w - self.rim * 2
        well = box(
            (inner_d, inner_w, 0.012),
            location=(-d / 2, w / 2, th - 0.004),
            name="shower_well",
        )
        assign(well, basin)
        parts.append(well)

        waste = cylinder(
            0.021,
            0.006,
            location=(-d / 2, w / 2, th + 0.001),
            name="shower_waste",
        )
        assign(waste, chrome)
        shade_smooth(waste)
        parts.append(waste)
        waste_hole = cylinder(
            0.012,
            0.004,
            location=(-d / 2, w / 2, th + 0.004),
            name="shower_waste_hole",
        )
        assign(
            waste_hole,
            solid_material(
                f"ShowerWasteHole_{self.factory_seed}", (0.06, 0.06, 0.06), roughness=0.7
            ),
        )
        parts.append(waste_hole)

        gz0 = th + 0.006
        gh = self.glass_h - gz0
        gz = gz0 + gh / 2
        p = self.post
        g_t = self.glass_t
        open_y = self.opening

        # Front screen at the room-facing side, walk-in gap near y=0.
        g_front_w = max(w - open_y - p, 0.22)
        g_front = box(
            (g_t, g_front_w, gh),
            location=(x_front - g_t / 2, w - p - g_front_w / 2, gz),
            name="shower_glass_front",
        )
        assign(g_front, glass)
        parts.append(g_front)

        # Return panel along the far side wall of the cubicle.
        g_side = box(
            (d - p * 1.2, g_t, gh),
            location=(-d / 2 + p * 0.2, w - g_t / 2, gz),
            name="shower_glass_side",
        )
        assign(g_side, glass)
        parts.append(g_side)

        def post_at(x, y):
            obj = box((p, p, self.glass_h), location=(x, y, self.glass_h / 2), name="shower_post")
            assign(obj, chrome)
            butil.modify_mesh(obj, "BEVEL", width=0.002, segments=2)
            shade_smooth(obj)
            return obj

        parts.append(post_at(x_front - p / 2, w - p / 2))
        parts.append(post_at(x_front - p / 2, open_y + p / 2))
        parts.append(post_at(x_wall + p / 2, w - p / 2))

        rail_f = box(
            (p * 0.65, g_front_w + p, p * 0.65),
            location=(x_front - p / 2, w - p - g_front_w / 2, self.glass_h - p * 0.3),
            name="shower_rail_front",
        )
        rail_s = box(
            (d - p * 0.4, p * 0.65, p * 0.65),
            location=(-d / 2 + p * 0.1, w - p / 2, self.glass_h - p * 0.3),
            name="shower_rail_side",
        )
        assign(rail_f, chrome)
        assign(rail_s, chrome)
        parts.extend([rail_f, rail_s])

        handle = cylinder(
            0.008,
            0.22,
            location=(x_front + 0.018, open_y + p + 0.06, 1.05),
            name="shower_handle",
        )
        assign(handle, chrome)
        shade_smooth(handle)
        parts.append(handle)
        for hz in (0.94, 1.16):
            stub = cylinder(
                0.007,
                0.028,
                location=(x_front + 0.004, open_y + p + 0.06, hz),
                rotation=(0, np.pi / 2, 0),
                name="shower_handle_stub",
            )
            assign(stub, chrome)
            shade_smooth(stub)
            parts.append(stub)

        # Mixer on the tiled back wall (~1100 mm AFF). Back face of the plate
        # sits on the placeholder's min-X so flush_fixture leaves no gap.
        plate_t = 0.012
        mix_x = x_wall + plate_t / 2
        mix_y = w * 0.42
        mix_plate = cylinder(
            0.055,
            plate_t,
            location=(mix_x, mix_y, self.mixer_z),
            rotation=(0, np.pi / 2, 0),
            name="shower_mixer_plate",
        )
        assign(mix_plate, chrome)
        shade_smooth(mix_plate)
        parts.append(mix_plate)
        mix_body = cylinder(
            0.028,
            0.022,
            location=(mix_x + 0.016, mix_y, self.mixer_z),
            rotation=(0, np.pi / 2, 0),
            name="shower_mixer_body",
        )
        assign(mix_body, chrome)
        shade_smooth(mix_body)
        parts.append(mix_body)
        lever = box(
            (0.012, 0.014, 0.09),
            location=(mix_x + 0.034, mix_y, self.mixer_z + 0.04),
            name="shower_lever",
        )
        assign(lever, chrome)
        butil.modify_mesh(lever, "BEVEL", width=0.002, segments=2)
        parts.append(lever)

        # Wall arm + rain head (~2050 mm AFF), projecting into the stall.
        arm_y = w * 0.50
        arm = cylinder(
            0.010,
            self.arm_len,
            location=(x_wall + 0.02 + self.arm_len / 2, arm_y, self.head_z),
            rotation=(0, np.pi / 2, 0),
            name="shower_arm",
        )
        assign(arm, chrome)
        shade_smooth(arm)
        parts.append(arm)
        flange = cylinder(
            0.024,
            plate_t,
            location=(x_wall + plate_t / 2, arm_y, self.head_z),
            rotation=(0, np.pi / 2, 0),
            name="shower_flange",
        )
        assign(flange, chrome)
        shade_smooth(flange)
        parts.append(flange)
        head_x = x_wall + 0.02 + self.arm_len
        head = cylinder(
            self.head_r,
            0.016,
            location=(head_x, arm_y, self.head_z - 0.008),
            name="shower_head",
        )
        assign(head, chrome)
        shade_smooth(head)
        parts.append(head)
        face = cylinder(
            self.head_r * 0.90,
            0.004,
            location=(head_x, arm_y, self.head_z - 0.018),
            name="shower_face",
        )
        assign(face, face_mat)
        parts.append(face)
        for i in range(8):
            ang = i * (2 * np.pi / 8)
            rr = self.head_r * 0.55
            hole = cylinder(
                0.004,
                0.005,
                location=(
                    head_x,
                    arm_y + rr * np.cos(ang),
                    self.head_z - 0.021,
                ),
                name="shower_nozzle",
            )
            assign(hole, face_mat)
            parts.append(hole)

        # Handheld on a slide rail — typical apartment fit-out.
        cap_t = 0.012
        cap_x = x_wall + cap_t / 2
        rail_x = x_wall + 0.018
        rail_y = w * 0.18
        slider = cylinder(
            0.009,
            0.72,
            location=(rail_x, rail_y, 1.28),
            name="shower_slider",
        )
        assign(slider, chrome)
        shade_smooth(slider)
        parts.append(slider)
        for zz in (0.92, 1.64):
            cap = cylinder(
                0.016,
                cap_t,
                location=(cap_x, rail_y, zz),
                rotation=(0, np.pi / 2, 0),
                name="shower_slider_cap",
            )
            assign(cap, chrome)
            shade_smooth(cap)
            parts.append(cap)
        hand = cylinder(
            0.022,
            0.085,
            location=(rail_x + 0.05, rail_y, 1.22),
            rotation=(0.45, np.pi / 2, 0),
            name="shower_handheld",
        )
        assign(hand, chrome)
        shade_smooth(hand)
        parts.append(hand)

        obj = join_objects(parts)
        obj.name = f"ShowerStallFactory({self.factory_seed}).stall"
        return obj
