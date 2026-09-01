import logging

import bpy
import numpy as np
from mathutils import Vector

from infinigen.assets.utils.object import new_bbox, new_plane
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_canonical_surfaces
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, int_hash

logger = logging.getLogger(__name__)

_SCENE_SUBSTRATE = None
_SCENE_SUBSTRATE_KEY = None


def _generation_seed():
    try:
        import gin

        return int(gin.query_parameter("OVERALL_SEED"))
    except Exception:
        return 0


def sample_peel_substrate(seed: int):
    """Exposed plaster/concrete/paint under the film. One draw per apartment."""
    with FixedSeed(seed):
        kind = np.random.choice(
            ["tan", "concrete", "white", "beige", "cool_grey"],
            p=[0.22, 0.30, 0.22, 0.16, 0.10],
        )
        if kind == "tan":
            plaster_lo = (
                np.random.uniform(0.34, 0.42),
                np.random.uniform(0.32, 0.39),
                np.random.uniform(0.26, 0.34),
                1.0,
            )
            plaster_hi = (
                np.random.uniform(0.50, 0.58),
                np.random.uniform(0.46, 0.54),
                np.random.uniform(0.40, 0.48),
                1.0,
            )
        elif kind == "concrete":
            g0 = np.random.uniform(0.36, 0.46)
            g1 = np.random.uniform(0.56, 0.68)
            plaster_lo = (g0 * 0.98, g0, g0 * 1.02, 1.0)
            plaster_hi = (g1 * 0.99, g1, g1 * 1.01, 1.0)
        elif kind == "white":
            g0 = np.random.uniform(0.68, 0.78)
            g1 = np.random.uniform(0.86, 0.95)
            plaster_lo = (g0, g0 * 0.995, g0 * 0.98, 1.0)
            plaster_hi = (g1, g1 * 0.998, g1 * 0.99, 1.0)
        elif kind == "beige":
            plaster_lo = (
                np.random.uniform(0.52, 0.62),
                np.random.uniform(0.48, 0.58),
                np.random.uniform(0.40, 0.50),
                1.0,
            )
            plaster_hi = (
                np.random.uniform(0.72, 0.82),
                np.random.uniform(0.68, 0.78),
                np.random.uniform(0.58, 0.68),
                1.0,
            )
        else:
            g0 = np.random.uniform(0.40, 0.50)
            g1 = np.random.uniform(0.60, 0.72)
            plaster_lo = (g0 * 0.96, g0, g0 * 1.04, 1.0)
            plaster_hi = (g1 * 0.97, g1, g1 * 1.03, 1.0)
        pit_tint = tuple(min(c * 0.55, 1.0) for c in plaster_lo[:3]) + (1.0,)
        flake_v = np.random.uniform(0.86, 0.96)
        flake_color = (flake_v, flake_v * 0.99, flake_v * 0.97, 1.0)
    return plaster_lo, plaster_hi, pit_tint, flake_color


def scene_peel_substrate():
    """Same exposed color for every peel in this apartment / generation seed."""
    global _SCENE_SUBSTRATE, _SCENE_SUBSTRATE_KEY
    key = _generation_seed()
    if _SCENE_SUBSTRATE is not None and _SCENE_SUBSTRATE_KEY == key:
        return _SCENE_SUBSTRATE
    _SCENE_SUBSTRATE = sample_peel_substrate(
        int_hash((key, "paint_peel_substrate"))
    )
    _SCENE_SUBSTRATE_KEY = key
    return _SCENE_SUBSTRATE


# Generated UV axes after the wall/ceiling plane is applied.
# Wall: YZ face; ceiling: XY face.
_AXES = {"wall": (1, 2), "ceiling": (0, 1)}


def _disable_peel_lighting(obj: bpy.types.Object) -> None:
    """Peel is a film on the plaster, not a card that shades the wall."""
    obj.visible_shadow = False
    obj.visible_diffuse = False
    if hasattr(obj, "visible_glossy"):
        obj.visible_glossy = False
    if hasattr(obj, "visible_transmission"):
        obj.visible_transmission = False
    if hasattr(obj, "visible_volume_scatter"):
        obj.visible_volume_scatter = False


def _fit_radius(cx, cy, rx, limit=0.27):
    """Keep |centre| + radius inside the plane so flakes never clip the border."""
    return min(rx, max(0.004, limit - max(abs(cx), abs(cy))))


def _ellipse_spec(cx, cy, rx, mode="add"):
    rx = _fit_radius(cx, cy, rx)
    ry = _fit_radius(cx, cy, rx * float(np.random.uniform(0.45, 1.70)))
    return {
        "cx": float(cx),
        "cy": float(cy),
        "rx": float(max(rx, 0.004)),
        "ry": float(max(ry, 0.004)),
        "rot": float(np.random.uniform(0.0, 2.0 * np.pi)),
        "mode": mode,
    }


def _sample_peel_layout():
    """Explicit flake layouts, same idea as crack styles. Always centred."""
    style = str(
        np.random.choice(
            ["chip_trail", "islands", "torn_patch", "scattered"],
            p=[0.30, 0.28, 0.24, 0.18],
        )
    )
    specs = []
    cx0 = float(np.random.uniform(-0.05, 0.05))
    cy0 = float(np.random.uniform(-0.05, 0.05))

    def add_blob(cx, cy, rx, n_lobes=0, n_bites=0):
        specs.append(_ellipse_spec(cx, cy, rx, "add"))
        parent = specs[-1]
        for _ in range(n_lobes):
            ang = float(np.random.uniform(0.0, 2.0 * np.pi))
            dist = parent["rx"] * float(np.random.uniform(0.35, 0.85))
            specs.append(
                _ellipse_spec(
                    parent["cx"] + dist * np.cos(ang),
                    parent["cy"] + dist * np.sin(ang),
                    parent["rx"] * float(np.random.uniform(0.35, 0.75)),
                    "add",
                )
            )
        for _ in range(n_bites):
            ang = float(np.random.uniform(0.0, 2.0 * np.pi))
            dist = parent["rx"] * float(np.random.uniform(0.70, 1.05))
            specs.append(
                _ellipse_spec(
                    parent["cx"] + dist * np.cos(ang),
                    parent["cy"] + dist * np.sin(ang),
                    parent["rx"] * float(np.random.uniform(0.18, 0.48)),
                    "sub",
                )
            )

    if style == "chip_trail":
        # Horizontal scrape of mixed chips (finger-scale up to a few cm).
        axis = float(np.random.uniform(0.0, 2.0 * np.pi))
        n_chips = int(np.random.randint(6, 13))
        half = float(np.random.uniform(0.09, 0.16))
        for i in range(n_chips):
            t = float(np.random.uniform(-half, half))
            jx = float(np.random.uniform(-0.025, 0.025))
            jy = float(np.random.uniform(-0.025, 0.025))
            cx = cx0 + t * np.cos(axis) + jx
            cy = cy0 + t * np.sin(axis) + jy
            rx = float(
                np.random.uniform(0.028, 0.055)
                if i == 0
                else np.random.uniform(0.006, 0.028)
            )
            add_blob(cx, cy, rx, n_lobes=int(np.random.rand() < 0.35), n_bites=0)
    elif style == "islands":
        n_main = int(np.random.randint(2, 5))
        for i in range(n_main):
            ang = float(np.random.uniform(0.0, 2.0 * np.pi))
            dist = 0.0 if i == 0 else float(np.random.uniform(0.04, 0.12))
            cx = cx0 + dist * np.cos(ang)
            cy = cy0 + dist * np.sin(ang)
            rx = float(
                np.random.uniform(0.055, 0.12)
                if i == 0
                else np.random.uniform(0.018, 0.055)
            )
            add_blob(
                cx,
                cy,
                rx,
                n_lobes=int(np.random.randint(1, 3)),
                n_bites=int(np.random.randint(1, 3)),
            )
    elif style == "torn_patch":
        add_blob(
            cx0,
            cy0,
            float(np.random.uniform(0.09, 0.17)),
            n_lobes=int(np.random.randint(2, 4)),
            n_bites=int(np.random.randint(2, 5)),
        )
        n_sat = int(np.random.randint(3, 8))
        for _ in range(n_sat):
            ang = float(np.random.uniform(0.0, 2.0 * np.pi))
            dist = float(np.random.uniform(0.06, 0.16))
            add_blob(
                cx0 + dist * np.cos(ang),
                cy0 + dist * np.sin(ang),
                float(np.random.uniform(0.007, 0.028)),
                n_lobes=0,
                n_bites=int(np.random.rand() < 0.4),
            )
    else:  # scattered mixed sizes
        n_main = int(np.random.randint(3, 7))
        for i in range(n_main):
            ang = float(np.random.uniform(0.0, 2.0 * np.pi))
            dist = 0.0 if i == 0 else float(np.random.uniform(0.03, 0.13))
            rx = float(
                np.random.uniform(0.04, 0.10)
                if i == 0
                else np.random.uniform(0.008, 0.045)
            )
            add_blob(
                cx0 + dist * np.cos(ang),
                cy0 + dist * np.sin(ang),
                rx,
                n_lobes=int(np.random.rand() < 0.55),
                n_bites=int(np.random.rand() < 0.55),
            )

    n_specks = int(np.random.randint(2, 9))
    for _ in range(n_specks):
        ang = float(np.random.uniform(0.0, 2.0 * np.pi))
        dist = float(np.random.uniform(0.02, 0.14))
        add_blob(
            cx0 + dist * np.cos(ang),
            cy0 + dist * np.sin(ang),
            float(np.random.uniform(0.004, 0.012)),
        )
    return style, specs


def create_paint_peel_material(
    name: str, seed: int, orientation: str = "wall"
) -> bpy.types.Material:
    """Centered delamination: torn-paper flakes, not a smooth ellipse.

    Intact paint is transparent so the real wall shows through. Peeled patches
    are a grainy plaster substrate with a millimetre paint lip. Substrate tint
    is scene-wide; flake layout is per instance.
    """
    plaster_lo, plaster_hi, pit_tint, flake_color = scene_peel_substrate()
    u_idx, v_idx = _AXES[orientation]
    with FixedSeed(seed):
        _, peel_specs = _sample_peel_layout()
        # Three-scale outline: coarse lobes, mid bays, fine brittle teeth.
        warp_scale = np.random.uniform(3.5, 7.5)
        warp_amp = np.random.uniform(0.028, 0.070)
        mid_scale = np.random.uniform(16.0, 38.0)
        mid_amp = np.random.uniform(0.018, 0.045)
        teeth_scale = np.random.uniform(48.0, 110.0)
        teeth_amp = np.random.uniform(0.10, 0.28)
        ridge_scale = np.random.uniform(22.0, 55.0)
        ridge_amp = np.random.uniform(0.08, 0.22)
        cell_scale = np.random.uniform(14.0, 36.0)
        cell_amp = np.random.uniform(0.06, 0.18)
        island_scale = np.random.uniform(12.0, 28.0)
        island_thr = np.random.uniform(0.74, 0.90)
        island_amt = float(
            0.0 if np.random.rand() < 0.28 else np.random.uniform(0.55, 1.0)
        )
        mapping_offset = tuple(np.random.uniform(0, 80, 3))
        grain_scale = np.random.uniform(70.0, 140.0)
        body_scale = np.random.uniform(14.0, 28.0)
        pit_scale = np.random.uniform(28.0, 55.0)
        # Paint-film thickness, not the old Distance~20 crater.
        rim_strength = np.random.uniform(0.35, 0.80)
        rim_distance = np.random.uniform(0.0014, 0.0036)
        plaster_bump = np.random.uniform(0.22, 0.42)

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

    def ramp(value, x0, x1, y0=0.0, y1=1.0, interp="LINEAR"):
        mr = node("ShaderNodeMapRange", interpolation_type=interp, clamp=True)
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

    def noise(vector, scale, detail=6.0, roughness=0.55, ntype="FBM"):
        n = node("ShaderNodeTexNoise", noise_dimensions="3D", noise_type=ntype)
        if hasattr(n, "normalize"):
            n.normalize = True
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = roughness
        links.new(vector, n.inputs["Vector"])
        return n.outputs["Fac"]

    node_output = node("ShaderNodeOutputMaterial")
    tex = node("ShaderNodeTexCoord")
    # Generated is 0–1 across the mesh AABB, so flakes stay centred on any size plane.
    sep = node("ShaderNodeSeparateXYZ")
    links.new(tex.outputs["Generated"], sep.inputs["Vector"])
    u = math("SUBTRACT", sep.outputs[u_idx], 0.5)
    v = math("SUBTRACT", sep.outputs[v_idx], 0.5)

    pattern_map = node("ShaderNodeMapping", vector_type="POINT")
    for i in range(3):
        pattern_map.inputs["Location"].default_value[i] = mapping_offset[i]
    links.new(tex.outputs["Object"], pattern_map.inputs["Vector"])
    pattern = pattern_map.outputs["Vector"]

    uv = node("ShaderNodeCombineXYZ")
    links.new(u, uv.inputs["X"])
    links.new(v, uv.inputs["Y"])
    uvv = uv.outputs["Vector"]

    # Domain warp: coarse lobes + mid-scale bays (torn paper, not a wobble).
    wu = math("MULTIPLY", math("SUBTRACT", noise(uvv, warp_scale, 5.0, 0.62), 0.5), warp_amp * 2.0)
    wv = math("MULTIPLY", math("SUBTRACT", noise(pattern, warp_scale * 1.37, 5.0, 0.58), 0.5), warp_amp * 2.0)
    mu = math("MULTIPLY", math("SUBTRACT", noise(uvv, mid_scale, 6.0, 0.70), 0.5), mid_amp * 2.0)
    mv = math("MULTIPLY", math("SUBTRACT", noise(pattern, mid_scale * 1.21, 6.0, 0.66), 0.5), mid_amp * 2.0)
    u_w = math("ADD", math("ADD", u, wu), mu)
    v_w = math("ADD", math("ADD", v, wv), mv)

    teeth = math(
        "MULTIPLY",
        math("SUBTRACT", noise(uvv, teeth_scale, 8.0, 0.78), 0.5),
        teeth_amp,
    )
    ridge = math(
        "MULTIPLY",
        math("SUBTRACT", noise(pattern, ridge_scale, 7.0, 0.72, "RIDGED_MULTIFRACTAL"), 0.5),
        ridge_amp,
    )
    cells = node("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="DISTANCE_TO_EDGE")
    cells.inputs["Scale"].default_value = cell_scale
    cells.inputs["Randomness"].default_value = 1.0
    links.new(uvv, cells.inputs["Vector"])
    cell_jag = math("MULTIPLY", math("SUBTRACT", 0.12, cells.outputs["Distance"]), cell_amp)
    jag = math("ADD", math("ADD", teeth, ridge), cell_jag)

    def flake_mask(spec):
        """Jagged ellipse: SDF + multi-scale torn-paper offset."""
        cu_ = math("SUBTRACT", u_w, spec["cx"])
        cv_ = math("SUBTRACT", v_w, spec["cy"])
        c, s = float(np.cos(spec["rot"])), float(np.sin(spec["rot"]))
        ru = math("ADD", math("MULTIPLY", cu_, c), math("MULTIPLY", cv_, s))
        rv = math("ADD", math("MULTIPLY", cu_, -s), math("MULTIPLY", cv_, c))
        eu = math("DIVIDE", ru, max(spec["rx"], 1e-4))
        ev = math("DIVIDE", rv, max(spec["ry"], 1e-4))
        edist = math(
            "POWER",
            math("ADD", math("MULTIPLY", eu, eu), math("MULTIPLY", ev, ev)),
            0.5,
        )
        return ramp(math("ADD", edist, jag), 0.86, 1.06, 1.0, 0.0, interp="LINEAR")

    add_specs = [s for s in peel_specs if s["mode"] == "add"]
    sub_specs = [s for s in peel_specs if s["mode"] == "sub"]
    peel = flake_mask(add_specs[0])
    for spec in add_specs[1:]:
        peel = math("MAXIMUM", peel, flake_mask(spec), clamp=True)
    for spec in sub_specs:
        bite = flake_mask(spec)
        peel = math("MULTIPLY", peel, math("SUBTRACT", 1.0, math("MULTIPLY", bite, 0.92)), clamp=True)

    if island_amt > 0.01:
        isl = noise(pattern, island_scale, 6.0, 0.62)
        deep = ramp(peel, 0.55, 0.92, 0.0, 1.0, interp="SMOOTHSTEP")
        holes = math(
            "MULTIPLY",
            ramp(isl, island_thr, min(island_thr + 0.07, 0.98), 0.0, 1.0),
            math("MULTIPLY", deep, island_amt),
            clamp=True,
        )
        peel = math("MULTIPLY", peel, math("SUBTRACT", 1.0, holes), clamp=True)

    # Box falloff so flakes die before any of the four plane edges.
    keep = math(
        "MULTIPLY",
        ramp(math("ABSOLUTE", u), 0.30, 0.42, 1.0, 0.0, interp="SMOOTHSTEP"),
        ramp(math("ABSOLUTE", v), 0.30, 0.42, 1.0, 0.0, interp="SMOOTHSTEP"),
        clamp=True,
    )
    peel = math("MULTIPLY", peel, keep, clamp=True)
    peel_ramp_out = peel

    # Lip / curl ring: peaks at the paint–substrate boundary.
    one_minus = nodes.new("ShaderNodeMath")
    one_minus.operation = "SUBTRACT"
    one_minus.inputs[0].default_value = 1.0
    links.new(peel_ramp_out, one_minus.inputs[1])

    edge = nodes.new("ShaderNodeMath")
    edge.operation = "MULTIPLY"
    links.new(peel_ramp_out, edge.inputs[0])
    links.new(one_minus.outputs["Value"], edge.inputs[1])

    edge_amp = nodes.new("ShaderNodeMath")
    edge_amp.operation = "MULTIPLY"
    edge_amp.inputs[1].default_value = 5.0
    links.new(edge.outputs["Value"], edge_amp.inputs[0])

    # --- Substrate: sandy plaster (body + fine grain + pits) ---
    n_body = nodes.new("ShaderNodeTexNoise")
    n_body.noise_dimensions = "3D"
    n_body.normalize = True
    n_body.noise_type = "FBM"
    n_body.inputs["Scale"].default_value = body_scale
    n_body.inputs["Detail"].default_value = 14.0
    n_body.inputs["Roughness"].default_value = 0.68
    links.new(tex.outputs["Object"], n_body.inputs["Vector"])

    n_grain = nodes.new("ShaderNodeTexNoise")
    n_grain.noise_dimensions = "3D"
    n_grain.normalize = True
    n_grain.noise_type = "FBM"
    n_grain.inputs["Scale"].default_value = grain_scale
    n_grain.inputs["Detail"].default_value = 12.0
    n_grain.inputs["Roughness"].default_value = 0.8
    links.new(tex.outputs["Object"], n_grain.inputs["Vector"])

    n_pits = nodes.new("ShaderNodeTexVoronoi")
    n_pits.voronoi_dimensions = "3D"
    n_pits.feature = "F1"
    n_pits.inputs["Scale"].default_value = pit_scale
    n_pits.inputs["Randomness"].default_value = 1.0
    links.new(tex.outputs["Object"], n_pits.inputs["Vector"])

    plaster_ramp = nodes.new("ShaderNodeValToRGB")
    plaster_ramp.color_ramp.elements[0].position = 0.15
    plaster_ramp.color_ramp.elements[1].position = 0.85
    plaster_ramp.color_ramp.elements[0].color = plaster_lo
    plaster_ramp.color_ramp.elements[1].color = plaster_hi
    links.new(n_body.outputs["Fac"], plaster_ramp.inputs["Fac"])

    grain_mix = nodes.new("ShaderNodeMixRGB")
    grain_mix.blend_type = "MIX"
    grain_mix.inputs["Fac"].default_value = 0.12
    links.new(plaster_ramp.outputs["Color"], grain_mix.inputs["Color1"])
    links.new(n_grain.outputs["Color"], grain_mix.inputs["Color2"])

    pit_ramp = nodes.new("ShaderNodeValToRGB")
    pit_ramp.color_ramp.elements[0].position = 0.0
    pit_ramp.color_ramp.elements[1].position = 0.45
    pit_ramp.color_ramp.elements[0].color = pit_tint
    pit_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    links.new(n_pits.outputs["Distance"], pit_ramp.inputs["Fac"])

    plaster_col = nodes.new("ShaderNodeMixRGB")
    plaster_col.blend_type = "MULTIPLY"
    plaster_col.inputs["Fac"].default_value = 0.22
    links.new(grain_mix.outputs["Color"], plaster_col.inputs["Color1"])
    links.new(pit_ramp.outputs["Color"], plaster_col.inputs["Color2"])

    # Soft crease under the flake (grey, not black).
    crease = nodes.new("ShaderNodeMixRGB")
    crease.blend_type = "MULTIPLY"
    crease.inputs["Fac"].default_value = 0.18
    crease.inputs["Color2"].default_value = (0.62, 0.58, 0.52, 1.0)
    links.new(plaster_col.outputs["Color"], crease.inputs["Color1"])
    links.new(edge_amp.outputs["Value"], crease.inputs["Fac"])

    paint_mix = nodes.new("ShaderNodeMixRGB")
    paint_mix.blend_type = "MIX"
    paint_mix.inputs["Color2"].default_value = flake_color
    links.new(edge_amp.outputs["Value"], paint_mix.inputs["Fac"])
    links.new(crease.outputs["Color"], paint_mix.inputs["Color1"])

    # Height: plaster grain, then a millimetre-scale raised lip (not Distance=20).
    grain_h = nodes.new("ShaderNodeMath")
    grain_h.operation = "MULTIPLY"
    grain_h.inputs[1].default_value = 0.45
    links.new(n_grain.outputs["Fac"], grain_h.inputs[0])

    body_h = nodes.new("ShaderNodeMath")
    body_h.operation = "MULTIPLY"
    body_h.inputs[1].default_value = 0.25
    links.new(n_body.outputs["Fac"], body_h.inputs[0])

    pit_h = nodes.new("ShaderNodeMath")
    pit_h.operation = "MULTIPLY"
    pit_h.inputs[1].default_value = 0.2
    links.new(n_pits.outputs["Distance"], pit_h.inputs[0])

    h1 = nodes.new("ShaderNodeMath")
    h1.operation = "ADD"
    links.new(grain_h.outputs["Value"], h1.inputs[0])
    links.new(body_h.outputs["Value"], h1.inputs[1])

    h2 = nodes.new("ShaderNodeMath")
    h2.operation = "ADD"
    links.new(h1.outputs["Value"], h2.inputs[0])
    links.new(pit_h.outputs["Value"], h2.inputs[1])

    bump_plaster = nodes.new("ShaderNodeBump")
    bump_plaster.inputs["Strength"].default_value = plaster_bump
    bump_plaster.inputs["Distance"].default_value = 0.0025
    links.new(h2.outputs["Value"], bump_plaster.inputs["Height"])

    # Millimetre paint-film step at the torn edge — not a 20 m crater.
    bump_lip = nodes.new("ShaderNodeBump")
    bump_lip.invert = True
    bump_lip.inputs["Strength"].default_value = rim_strength
    bump_lip.inputs["Distance"].default_value = rim_distance
    links.new(peel_ramp_out, bump_lip.inputs["Height"])
    links.new(bump_plaster.outputs["Normal"], bump_lip.inputs["Normal"])

    rough = nodes.new("ShaderNodeMath")
    rough.operation = "MULTIPLY"
    rough.use_clamp = True
    rough.inputs[1].default_value = 0.35
    links.new(edge_amp.outputs["Value"], rough.inputs[0])

    rough_sub = nodes.new("ShaderNodeMath")
    rough_sub.operation = "SUBTRACT"
    rough_sub.use_clamp = True
    rough_sub.inputs[0].default_value = 0.92
    links.new(rough.outputs["Value"], rough_sub.inputs[1])

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(paint_mix.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(rough_sub.outputs["Value"], bsdf.inputs["Roughness"])
    links.new(bump_lip.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(peel_ramp_out, bsdf.inputs["Alpha"])
    spec_key = (
        "Specular IOR Level" if "Specular IOR Level" in bsdf.inputs else "Specular"
    )
    if spec_key in bsdf.inputs:
        bsdf.inputs[spec_key].default_value = 0.12
    links.new(bsdf.outputs["BSDF"], node_output.inputs["Surface"])

    mat.blend_method = "CLIP"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"
    if hasattr(mat, "alpha_threshold"):
        mat.alpha_threshold = 0.42
    return mat


class PaintPeelPlaneFactory(AssetFactory):
    """Procedural wall-mounted paint peeling plane (centered, sparse, non-metallic)."""

    def __init__(self, factory_seed, coarse: bool = False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.5, 1.5)

    def create_placeholder(self, **kwargs):
        # Same geometry/orientation as crack planes: thin vertical bbox
        ph = new_bbox(-0.005, 0.005, -0.5, 0.5, -0.5, 0.5)
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        plane = new_plane()

        with FixedSeed(int_hash((self.factory_seed, kwargs.get("i", 0), "geom"))):
            scale_z_val = np.random.uniform(0.6, 1.0) * self.plane_size / 2
            scale_y_val = np.random.uniform(0.6, 1.0) * self.plane_size / 2

        plane.scale = (scale_z_val, scale_y_val, 1)
        plane.rotation_euler = (0.0, np.pi / 2, 0.0)
        butil.apply_transform(plane, loc=False, rot=True, scale=True)

        mat = create_paint_peel_material(
            name=f"PaintPeelMaterial_{id(plane)}",
            seed=int_hash((self.factory_seed, kwargs.get("i", 0))),
            orientation="wall",
        )
        plane.data.materials.append(mat)
        _disable_peel_lighting(plane)
        return plane

    def finalize_assets(self, assets):
        """Embed paint-peel planes slightly into the wall, same logic as crack planes."""
        EMBED_OFFSET = -0.000226
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
                logger.warning("Failed to embed paint peel plane %s: %s", obj.name, e)


class CeilingPeelFactory(AssetFactory):
    """Same peel material as walls, hung flush to the ceiling (Top vs ceiling).

    Name is short on purpose: Blender object names cap at 63 chars, and
    ``CeilingPaintPeelPlaneFactory(...).spawn_placeholder(...)`` was truncated
    so populate could not parse the instance seed.
    """

    def __init__(self, factory_seed, coarse: bool = False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.5, 1.5)

    def create_placeholder(self, **kwargs):
        ph = new_bbox(-0.5, 0.5, -0.5, 0.5, -0.005, 0.005)
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        plane = new_plane()
        with FixedSeed(int_hash((self.factory_seed, kwargs.get("i", 0), "geom"))):
            sx = np.random.uniform(0.6, 1.0) * self.plane_size / 2
            sy = np.random.uniform(0.6, 1.0) * self.plane_size / 2
        plane.scale = (sx, sy, 1)
        butil.apply_transform(plane, loc=False, rot=True, scale=True)
        mat = create_paint_peel_material(
            name=f"PaintPeelMaterial_{id(plane)}",
            seed=int_hash((self.factory_seed, kwargs.get("i", 0))),
            orientation="ceiling",
        )
        plane.data.materials.append(mat)
        _disable_peel_lighting(plane)
        return plane

    def finalize_assets(self, assets):
        EMBED_OFFSET = -0.000226
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
                    top_poly = max(obj.data.polygons, key=lambda p: p.normal.z)
                ceil_normal = np.array(butil.global_polygon_normal(obj, top_poly))
                obj.location += Vector(ceil_normal * EMBED_OFFSET)
            except Exception as e:
                logger.warning(
                    "Failed to embed ceiling paint peel plane %s: %s", obj.name, e
                )
