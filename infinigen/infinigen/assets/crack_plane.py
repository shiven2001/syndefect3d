# Copyright (C) 2023, Princeton University.
# Procedural crack plane asset for Infinigen (wall-mounted, defect semantics).
# Adapted from defect_generation/procedural_crack_plane_gen.py.

import logging

import bpy
import numpy as np
from mathutils import Vector

from infinigen.assets.utils.object import new_bbox
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_canonical_surfaces
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, int_hash

logger = logging.getLogger(__name__)


# Sink unused (transparent) grid slightly into the wall so coplanar film
# cannot catch light or contact-shadow a rectangle onto the plaster.
EMBED_DEPTH = 0.0002

# Half-thickness of the placeholder bbox below. The solver seats the
# placeholder's BACK face against the wall, but the asset is built at the
# placeholder's centre, so it is born this far proud of the plaster.
PLACEHOLDER_HALF_T = 0.005

# Lift the channel floor to the wall surface inside the damage mask only.
MIN_PROUD = 0.00008

# Target quad size for the displaced grid. The channel is ~1 mm across, so the
# mesh has to resolve below that or true displacement has nothing to bite on.

TARGET_QUAD_M = 0.0012
MAX_GRID_SUBDIV = 640

# Wall planes face local +X (rotated), ceiling planes face local +Z.
# (u, v, rotation-axis) index triples into the Object coordinate.
_AXES = {"wall": (1, 2, 0), "ceiling": (0, 1, 2)}


def grid_subdivisions(extent_m: float) -> int:
    """Grid cuts needed to resolve a crack channel across `extent_m` of wall."""
    return int(np.clip(round(extent_m / TARGET_QUAD_M), 64, MAX_GRID_SUBDIV))


def create_crack_material(
    name: str, seed: int, orientation: str = "wall"
) -> bpy.types.Material:
    """Hairline plaster crack: a jagged run with forks, not a Voronoi mosaic.

    Real inspection photos are an irregular fissure that splits, tapers, and
    pinches, with a one-sided lifted paint lip. The main split is the zero-set
    of ``u - path(v)``; doglegs kink that run and explicit forks peel off it.
    Displacement and opacity live only on the damage so the rest of the plane
    is gone (no grey rectangle on the wall).
    """
    u_idx, v_idx, rot_idx = _AXES[orientation]

    with FixedSeed(seed):
        # Style drives topology; continuous params add per-instance variety.
        style = str(
            np.random.choice(
                [
                    "hairline",
                    "hairline_fork",
                    "moderate",
                    "branched",
                    "double_run",
                ],
                p=[0.32, 0.28, 0.22, 0.12, 0.06],
            )
        )

        if style == "hairline":
            w_min = np.random.uniform(0.00025, 0.00065)
            w_max = np.random.uniform(0.0010, 0.0035)
            n_kinks = int(np.random.randint(0, 2))
            n_forks = 0
            y_split_enabled = False
        elif style == "hairline_fork":
            w_min = np.random.uniform(0.00030, 0.00080)
            w_max = np.random.uniform(0.0015, 0.0045)
            n_kinks = int(np.random.randint(0, 2))
            n_forks = int(np.random.randint(1, 3))
            y_split_enabled = bool(np.random.rand() < 0.55)
        elif style == "moderate":
            w_min = np.random.uniform(0.00045, 0.00110)
            w_max = np.random.uniform(0.0025, 0.0075)
            n_kinks = int(np.random.randint(1, 3))
            n_forks = int(np.random.randint(2, 5))
            y_split_enabled = bool(np.random.rand() < 0.70)
        elif style == "branched":
            w_min = np.random.uniform(0.00055, 0.00140)
            w_max = np.random.uniform(0.0040, 0.0120)
            n_kinks = int(np.random.randint(1, 4))
            n_forks = int(np.random.randint(3, 7))
            y_split_enabled = True
        else:  # double_run — two independent subtle runs
            w_min = np.random.uniform(0.00025, 0.00075)
            w_max = np.random.uniform(0.0012, 0.0040)
            n_kinks = int(np.random.randint(0, 2))
            n_forks = int(np.random.randint(0, 2))
            y_split_enabled = False

        w_fork_floor = np.random.uniform(0.00025, 0.00075)
        w_lip = np.random.uniform(0.0006, 0.0020)
        h_lip = np.random.uniform(0.0004, 0.0014)
        crack_depth = np.random.uniform(0.0005, 0.0016)

        crack_angle = np.random.uniform(0, 2 * np.pi)
        crack_offset = np.random.uniform(-0.08, 0.08)
        lip_on_plus = bool(np.random.rand() > 0.5)
        lip_strength = float(np.random.uniform(0.35, 1.0))

        wander_freq = np.random.uniform(0.9, 4.5)
        wander_amp = np.random.uniform(0.05, 0.22)
        jag_freq = np.random.uniform(16.0, 72.0)
        jag_amp = np.random.uniform(0.004, 0.028)
        micro_freq = np.random.uniform(55.0, 180.0)
        micro_amp = np.random.uniform(0.0005, 0.0032)
        width_freq = np.random.uniform(3.5, 16.0)

        kink_specs = [
            {
                "v": float(np.random.uniform(-0.18, 0.18)),
                "slope": float(
                    np.random.uniform(0.15, 1.05)
                    * (1.0 if np.random.rand() > 0.5 else -1.0)
                ),
            }
            for _ in range(n_kinks)
        ]

        fork_specs = []
        if n_forks > 0 and y_split_enabled:
            y_split = float(np.random.uniform(-0.10, 0.10))
            y_slope = float(np.random.uniform(0.45, 2.8))
            y_dir = 1.0 if np.random.rand() > 0.5 else -1.0
            fork_specs.extend(
                [
                    {
                        "split": y_split,
                        "slope": y_slope,
                        "length": float(np.random.uniform(0.10, 0.38)),
                        "direction": y_dir,
                        "width_scale": float(np.random.uniform(0.35, 0.85)),
                    },
                    {
                        "split": y_split,
                        "slope": -y_slope * float(np.random.uniform(0.25, 0.85)),
                        "length": float(np.random.uniform(0.06, 0.24)),
                        "direction": y_dir,
                        "width_scale": float(np.random.uniform(0.20, 0.60)),
                    },
                ]
            )
            n_forks = max(n_forks - 2, 0)

        split_cursor = float(np.random.uniform(-0.12, 0.12))
        for _ in range(n_forks):
            split_cursor += float(np.random.uniform(0.04, 0.14)) * (
                1.0 if np.random.rand() > 0.5 else -1.0
            )
            fork_specs.append(
                {
                    "split": float(np.clip(split_cursor, -0.22, 0.22)),
                    "slope": float(
                        np.random.uniform(0.35, 2.6)
                        * (1.0 if np.random.rand() > 0.5 else -1.0)
                    ),
                    "length": float(np.random.uniform(0.05, 0.32)),
                    "direction": 1.0 if np.random.rand() > 0.5 else -1.0,
                    "width_scale": float(np.random.uniform(0.18, 0.75)),
                }
            )

        # Optional second hairline run (offset parallel path).
        second_run = None
        if style == "double_run":
            second_run = {
                "offset": float(np.random.uniform(0.012, 0.06)),
                "width_scale": float(np.random.uniform(0.55, 0.95)),
                "slope_delta": float(np.random.uniform(-0.35, 0.35)),
            }

        # Micro hairlines near the main split (sparse, very thin).
        micro_line_count = int(
            np.random.choice([0, 0, 1, 2, 3], p=[0.35, 0.25, 0.20, 0.12, 0.08])
        )
        micro_line_specs = [
            {
                "split": float(np.random.uniform(-0.16, 0.16)),
                "slope": float(np.random.uniform(-1.8, 1.8)),
                "length": float(np.random.uniform(0.03, 0.16)),
                "direction": 1.0 if np.random.rand() > 0.5 else -1.0,
            }
            for _ in range(micro_line_count)
        ]

        lift_freq = np.random.uniform(2.0, 9.0)
        lift_floor = np.random.uniform(0.08, 0.40)

        peel_chance = float(
            np.random.choice([0.0, 0.0, 0.0, 0.0, 0.45, 0.62], p=[0.22, 0.18, 0.15, 0.15, 0.18, 0.12])
        )
        peel_scale = np.random.uniform(8.0, 22.0)
        peel_h = np.random.uniform(0.0005, 0.0020)

        pv = np.random.uniform(0.70, 0.86)
        paint_col = (pv, pv * 0.995, pv * 0.978, 1.0)
        sv = min(pv * np.random.uniform(0.82, 0.94), 0.90)
        substrate_col = (sv, sv * 0.985, sv * 0.955, 1.0)
        dv = pv * np.random.uniform(0.04, 0.18)
        crack_dark = (dv, dv * 0.97, dv * 0.93, 1.0)
        paint_rough = np.random.uniform(0.52, 0.68)
        substrate_rough = np.random.uniform(0.88, 0.97)

        roller_freq = np.random.uniform(80.0, 160.0)
        roller_aniso = np.random.uniform(4.0, 9.0)
        roller_strength = np.random.uniform(0.06, 0.20)
        grain_freq = np.random.uniform(90.0, 180.0)
        grain_strength = np.random.uniform(0.02, 0.09)
        mapping_offset = tuple(np.random.uniform(0, 100, 3))

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    def node(kind, **props):
        n = nodes.new(kind)
        for k, val in props.items():
            setattr(n, k, val)
        return n

    def ramp(value, x0, x1, y0=0.0, y1=1.0):
        """Smoothstep `value` from (x0, x1) onto (y0, y1). Bounds may be sockets."""
        mr = node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP", clamp=True)
        for sock_name, operand in (
            ("From Min", x0),
            ("From Max", x1),
            ("To Min", y0),
            ("To Max", y1),
        ):
            if hasattr(operand, "is_linked"):
                links.new(operand, mr.inputs[sock_name])
            else:
                mr.inputs[sock_name].default_value = operand
        links.new(value, mr.inputs["Value"])
        return mr.outputs["Result"]

    def math(op, a, b=None, clamp=False):
        m = node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, operand in enumerate((a, b)):
            if operand is None:
                continue
            if hasattr(operand, "is_linked"):
                links.new(operand, m.inputs[i])
            else:
                m.inputs[i].default_value = operand
        return m.outputs["Value"]

    def noise(vector, scale, detail=6.0, roughness=0.5):
        n = node("ShaderNodeTexNoise", noise_dimensions="3D", noise_type="FBM")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = roughness
        links.new(vector, n.inputs["Vector"])
        return n.outputs["Fac"]

    # --- surface coordinate ---
    tex_coord = node("ShaderNodeTexCoord")
    mapping = node("ShaderNodeMapping", vector_type="POINT")
    mapping.inputs["Rotation"].default_value[rot_idx] = crack_angle
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    pattern_map = node("ShaderNodeMapping", vector_type="POINT")
    for i in range(3):
        pattern_map.inputs["Location"].default_value[i] = mapping_offset[i]
    links.new(mapping.outputs["Vector"], pattern_map.inputs["Vector"])
    pattern = pattern_map.outputs["Vector"]

    sep = node("ShaderNodeSeparateXYZ")
    links.new(mapping.outputs["Vector"], sep.inputs["Vector"])
    u = sep.outputs[u_idx]
    v = sep.outputs[v_idx]

    v_only = node("ShaderNodeCombineXYZ")
    links.new(v, v_only.inputs["X"])
    v_only.inputs["Y"].default_value = float(mapping_offset[0])
    v_only.inputs["Z"].default_value = float(mapping_offset[1])
    v_line = v_only.outputs["Vector"]

    meander = math(
        "MULTIPLY",
        math("SUBTRACT", noise(v_line, wander_freq, 5.0, 0.55), 0.5),
        wander_amp * 2.0,
    )
    jag = math(
        "MULTIPLY",
        math("SUBTRACT", noise(v_line, jag_freq, 8.0, 0.72), 0.5),
        jag_amp * 2.0,
    )
    micro = math(
        "MULTIPLY",
        math("SUBTRACT", noise(v_line, micro_freq, 4.0, 0.45), 0.5),
        micro_amp * 2.0,
    )
    path = math("ADD", math("ADD", math("ADD", meander, jag), micro), crack_offset)
    for kink in kink_specs:
        along_k = math("MAXIMUM", math("SUBTRACT", v, kink["v"]), 0.0)
        path = math("ADD", path, math("MULTIPLY", kink["slope"], along_k))
    signed = math("SUBTRACT", u, path)
    dist = math("ABSOLUTE", signed)

    width_n = noise(v_line, width_freq, 3.0, 0.45)
    # Mostly hairline, with rarer medium stretches and a few fat pockets.
    mid = ramp(width_n, 0.36, 0.54, 0.0, 0.35)
    fat = ramp(width_n, 0.52, 0.78, 0.0, 1.0)
    width_mod = math("ADD", mid, math("MULTIPLY", fat, fat))
    w_eff = math(
        "ADD",
        w_min,
        math("MULTIPLY", math("SUBTRACT", w_max, w_min), width_mod),
    )
    main_channel = ramp(dist, 0.0, w_eff, 1.0, 0.0)

    def fork_channel(split, slope, length, direction, width_scale):
        """A branch that leaves the main path at `split` and tapers out."""
        along = math(
            "MAXIMUM",
            math("MULTIPLY", math("SUBTRACT", v, split), direction),
            0.0,
        )
        fade = ramp(along, length * 0.55, length, 1.0, 0.0)
        path_f = math(
            "ADD",
            math("ADD", path, math("MULTIPLY", slope, along)),
            math("MULTIPLY", jag, 0.45),
        )
        dist_f = math("ABSOLUTE", math("SUBTRACT", u, path_f))
        taper = math(
            "ADD",
            0.12,
            math(
                "MULTIPLY",
                0.88,
                math("SUBTRACT", 1.0, math("DIVIDE", along, max(length, 1e-6))),
            ),
        )
        width_f = math(
            "MULTIPLY",
            math(
                "MAXIMUM",
                math("MULTIPLY", w_eff, width_scale),
                w_fork_floor,
            ),
            taper,
        )
        return math("MULTIPLY", ramp(dist_f, 0.0, width_f, 1.0, 0.0), fade)

    forks = None
    for spec in fork_specs:
        fc = fork_channel(
            spec["split"],
            spec["slope"],
            spec["length"],
            spec["direction"],
            spec["width_scale"],
        )
        forks = fc if forks is None else math("MAXIMUM", forks, fc, clamp=True)

    if second_run is not None:
        path2 = math("ADD", path, second_run["offset"])
        dist2 = math("ABSOLUTE", math("SUBTRACT", u, path2))
        w2 = math("MULTIPLY", w_eff, second_run["width_scale"])
        run2 = ramp(dist2, 0.0, w2, 1.0, 0.0)
        forks = (
            run2 if forks is None else math("MAXIMUM", forks, run2, clamp=True)
        )

    micro_hair = None
    for spec in micro_line_specs:
        along = math(
            "MAXIMUM",
            math("MULTIPLY", math("SUBTRACT", v, spec["split"]), spec["direction"]),
            0.0,
        )
        fade = ramp(along, spec["length"] * 0.45, spec["length"], 1.0, 0.0)
        path_m = math(
            "ADD",
            math("ADD", path, math("MULTIPLY", spec["slope"], along)),
            math("MULTIPLY", jag, 0.25),
        )
        dist_m = math("ABSOLUTE", math("SUBTRACT", u, path_m))
        w_m = w_fork_floor * 0.42
        line = math(
            "MULTIPLY",
            ramp(dist_m, 0.0, w_m, 1.0, 0.0),
            math("MULTIPLY", fade, ramp(dist, 0.0, w_max * 6.0, 1.0, 0.0)),
        )
        micro_hair = (
            line if micro_hair is None else math("MAXIMUM", micro_hair, line, clamp=True)
        )

    channel = main_channel
    if forks is not None:
        channel = math("MAXIMUM", channel, forks, clamp=True)
    if micro_hair is not None:
        channel = math("MAXIMUM", channel, micro_hair, clamp=True)

    peel_n = noise(pattern, peel_scale, 3.0, 0.4)
    peel = math(
        "MULTIPLY",
        ramp(peel_n, peel_chance, min(peel_chance + 0.08, 0.98)),
        ramp(dist, w_max, w_max * 8.0, 1.0, 0.0),
    )
    if peel_chance < 0.05:
        peel = math("MULTIPLY", peel, 0.0)

    exposed = math("MAXIMUM", channel, peel, clamp=True)
    covered = math("SUBTRACT", 1.0, exposed)

    lift_amt = ramp(noise(v_line, lift_freq, 3.0), 0.28, 0.72, lift_floor, 1.0)
    lip_profile = math(
        "MULTIPLY",
        ramp(dist, w_eff, math("ADD", w_eff, w_lip), 1.0, 0.0),
        covered,
    )
    if lip_on_plus:
        lip_side = ramp(signed, -0.00015, 0.00015, 0.0, 1.0)
    else:
        lip_side = ramp(signed, -0.00015, 0.00015, 1.0, 0.0)
    lip = math(
        "MULTIPLY",
        math("MULTIPLY", lip_profile, lip_side),
        math("MULTIPLY", lift_amt, lip_strength),
    )

    peel_curl = math(
        "MULTIPLY",
        math("MULTIPLY", peel, ramp(dist, 0.0, w_max * 2.0, 0.15, 1.0)),
        peel_h,
    )
    crack_body = channel
    lift = math(
        "MULTIPLY",
        ramp(crack_body, 0.06, 0.32),
        crack_depth + MIN_PROUD,
    )
    carve = math("MULTIPLY", main_channel, crack_depth)
    if forks is not None:
        carve = math("ADD", carve, math("MULTIPLY", forks, crack_depth * 0.55))
    if micro_hair is not None:
        carve = math("ADD", carve, math("MULTIPLY", micro_hair, crack_depth * 0.35))
    height = math(
        "ADD",
        math("SUBTRACT", peel_curl, carve),
        math("MULTIPLY", lip, h_lip),
    )
    height = math("ADD", height, lift)
    # Displacement only on damage. Unused film height = -EMBED_DEPTH (inside wall).
    disp_mask = math("MAXIMUM", math("MAXIMUM", crack_body, lip), peel, clamp=True)
    height = math(
        "ADD",
        math("MULTIPLY", height, disp_mask),
        math("MULTIPLY", math("SUBTRACT", 1.0, disp_mask), -EMBED_DEPTH),
    )
    disp = node("ShaderNodeDisplacement")
    disp.inputs["Midlevel"].default_value = 0.0
    disp.inputs["Scale"].default_value = 1.0
    links.new(height, disp.inputs["Height"])

    base_mix = node("ShaderNodeMixRGB", blend_type="MIX", use_clamp=True)
    base_mix.inputs["Color1"].default_value = paint_col
    base_mix.inputs["Color2"].default_value = substrate_col
    links.new(peel, base_mix.inputs["Fac"])
    base_col = node("ShaderNodeMixRGB", blend_type="MIX", use_clamp=True)
    base_col.inputs["Color2"].default_value = crack_dark
    links.new(base_mix.outputs["Color"], base_col.inputs["Color1"])
    links.new(channel, base_col.inputs["Fac"])

    rough = node("ShaderNodeMapRange", clamp=True)
    rough.inputs["From Min"].default_value = 0.0
    rough.inputs["From Max"].default_value = 1.0
    rough.inputs["To Min"].default_value = paint_rough
    rough.inputs["To Max"].default_value = substrate_rough
    links.new(exposed, rough.inputs["Value"])

    bsdf = node("ShaderNodeBsdfPrincipled")
    links.new(base_col.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])

    roller_map = node("ShaderNodeMapping", vector_type="POINT")
    roller_map.inputs["Scale"].default_value[u_idx] = roller_aniso
    links.new(pattern, roller_map.inputs["Vector"])
    roller_h = noise(roller_map.outputs["Vector"], roller_freq, 3.0, 0.4)
    grain_h = noise(pattern, grain_freq, 5.0, 0.55)
    paint_h = math("ADD", math("MULTIPLY", roller_h, roller_strength), math("MULTIPLY", grain_h, grain_strength))
    paint_bump = node("ShaderNodeBump")
    paint_bump.inputs["Strength"].default_value = 1.0
    paint_bump.inputs["Distance"].default_value = 0.0004
    links.new(paint_h, paint_bump.inputs["Height"])
    links.new(paint_bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Hard mask tied to displacement mask so alpha and geometry agree.
    alpha = math("GREATER_THAN", disp_mask, 0.10)

    trans = node("ShaderNodeBsdfTransparent")
    mix_sh = node("ShaderNodeMixShader")
    links.new(alpha, mix_sh.inputs["Fac"])
    links.new(trans.outputs["BSDF"], mix_sh.inputs[1])
    links.new(bsdf.outputs["BSDF"], mix_sh.inputs[2])
    if "Alpha" in bsdf.inputs:
        links.new(alpha, bsdf.inputs["Alpha"])

    out = node("ShaderNodeOutputMaterial")
    links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])
    links.new(disp.outputs["Displacement"], out.inputs["Displacement"])

    mat.displacement_method = "BOTH"
    mat.use_backface_culling = True
    if hasattr(mat, "show_transparent_back"):
        mat.show_transparent_back = False
    if hasattr(mat, "use_transparent_shadow"):
        mat.use_transparent_shadow = False
    if hasattr(mat, "blend_method"):
        mat.blend_method = "CLIP"
    if hasattr(mat, "alpha_threshold"):
        mat.alpha_threshold = 0.12
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"
    return mat


class CrackPlaneFactory(AssetFactory):
    """Procedural wall-mounted crack plane (hairline cracks). Uses same placement and scoring as defect planes."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        # Randomize max extent; push range up so cracks are not too tiny
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.5, 1.5)

    def create_placeholder(self, **kwargs):
        # Vertical thin box: thin in X (Back/Front), extent in Y/Z for wall-mounted plaque
        ph = new_bbox(
            -PLACEHOLDER_HALF_T, PLACEHOLDER_HALF_T, -0.5, 0.5, -0.5, 0.5
        )
        # tag_canonical_surfaces expects triangulated mesh (vert_mask_to_tri_mask uses 3 verts per poly)
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        # Required for constraint solver: Back/Front/Top/Bottom face tags for StableAgainst(back, wall)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        # Per-instance geometric variation (size only; no random rotation)
        with FixedSeed(int_hash((self.factory_seed, kwargs.get("i", 0), "geom"))):
            scale_z_val = (
                np.random.uniform(0.5, 1.0) * self.plane_size / 2
            )
            scale_y_val = (
                np.random.uniform(0.5, 1.0) * self.plane_size / 2
            )
            roll = float(np.random.uniform(-0.45, 0.45))

        cuts = grid_subdivisions(2 * max(scale_z_val, scale_y_val))
        bpy.ops.mesh.primitive_grid_add(
            x_subdivisions=cuts, y_subdivisions=cuts, size=2, location=(0, 0, 0)
        )
        plane = bpy.context.active_object
        butil.apply_transform(plane, loc=True)

        plane.scale = (scale_z_val, scale_y_val, 1)
        plane.rotation_euler = (0.0, np.pi / 2, roll)
        butil.apply_transform(plane, loc=False, rot=True, scale=True)
        mat = create_crack_material(
            name=f"CrackMaterial_{id(plane)}",
            seed=int_hash((self.factory_seed, kwargs.get("i", 0))),
            orientation="wall",
        )
        plane.data.materials.append(mat)
        plane.visible_shadow = False
        plane.visible_diffuse = False
        return plane

    def finalize_assets(self, assets):
        """Embed crack planes into the wall (same as StaticDefectPlaneFactory)."""
        # Seat the plane on the wall surface. The film has zero height there,
        # so it lies flush with no gap to cast a rectangle across; the damage
        # is lifted onto the plaster by the shader instead.
        EMBED_OFFSET = -(PLACEHOLDER_HALF_T + EMBED_DEPTH)
        for obj in assets:
            if obj.type != "MESH" or not obj.data.polygons:
                continue
            try:
                from infinigen.core import tags as t
                from infinigen.core.tagging import tagged_face_mask

                back_mask = tagged_face_mask(obj, {t.Subpart.Back})
                if back_mask.any():
                    back_faces = [i for i, tag in enumerate(back_mask) if tag]
                    if back_faces:
                        largest_back_face_idx = max(
                            back_faces, key=lambda idx: obj.data.polygons[idx].area
                        )
                        back_poly = obj.data.polygons[largest_back_face_idx]
                    else:
                        continue
                else:
                    back_poly = max(
                        obj.data.polygons,
                        key=lambda p: -p.normal.y if p.normal.y < 0 else -1e6,
                    )
                wall_normal = np.array(butil.global_polygon_normal(obj, back_poly))
                translation = Vector(wall_normal * EMBED_OFFSET)
                obj.location += translation
            except Exception as e:
                logger.warning("Failed to embed crack plane %s: %s", obj.name, e)


class CeilingCrackPlaneFactory(AssetFactory):
    """Same crack material as walls, hung flush to the ceiling (Top vs ceiling)."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.5, 1.5)

    def create_placeholder(self, **kwargs):
        # Horizontal thin box: Top/Bottom are the large faces for hanging.
        ph = new_bbox(
            -0.5, 0.5, -0.5, 0.5, -PLACEHOLDER_HALF_T, PLACEHOLDER_HALF_T
        )
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        with FixedSeed(int_hash((self.factory_seed, kwargs.get("i", 0), "geom"))):
            sx = np.random.uniform(0.5, 1.0) * self.plane_size / 2
            sy = np.random.uniform(0.5, 1.0) * self.plane_size / 2
            roll = float(np.random.uniform(-0.45, 0.45))
        cuts = grid_subdivisions(2 * max(sx, sy))
        bpy.ops.mesh.primitive_grid_add(
            x_subdivisions=cuts, y_subdivisions=cuts, size=2, location=(0, 0, 0)
        )
        plane = bpy.context.active_object
        butil.apply_transform(plane, loc=True)
        plane.scale = (sx, sy, 1)
        plane.rotation_euler = (0.0, 0.0, roll)
        butil.apply_transform(plane, loc=False, rot=True, scale=True)
        mat = create_crack_material(
            name=f"CrackMaterial_{id(plane)}",
            seed=int_hash((self.factory_seed, kwargs.get("i", 0))),
            orientation="ceiling",
        )
        plane.data.materials.append(mat)
        plane.visible_shadow = False
        plane.visible_diffuse = False
        return plane

    def finalize_assets(self, assets):
        # `top_poly`'s normal already points up at the ceiling, so this offset
        # is positive - negating it drove the plane down, away from the slab.
        EMBED_OFFSET = PLACEHOLDER_HALF_T
        for obj in assets:
            if obj.type != "MESH" or not obj.data.polygons:
                continue
            try:
                from infinigen.core import tags as t
                from infinigen.core.tagging import tagged_face_mask

                top_mask = tagged_face_mask(obj, {t.Subpart.Top})
                if top_mask.any():
                    top_faces = [i for i, tag in enumerate(top_mask) if tag]
                    top_poly = obj.data.polygons[
                        max(top_faces, key=lambda idx: obj.data.polygons[idx].area)
                    ]
                else:
                    top_poly = max(
                        obj.data.polygons,
                        key=lambda p: p.normal.z,
                    )
                ceil_normal = np.array(butil.global_polygon_normal(obj, top_poly))
                obj.location += Vector(ceil_normal * EMBED_OFFSET)
            except Exception as e:
                logger.warning(
                    "Failed to embed ceiling crack plane %s: %s", obj.name, e
                )
