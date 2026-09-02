#!/usr/bin/env python3
# Paint-blister (bubble) plane: clustered spherical-cap lifts of the wall film.
# Export material prefix BubbleMaterial_* (class 4).

import logging
import math

import bpy
import bmesh
import numpy as np
from mathutils import Vector, noise

from infinigen.assets.crack_plane import _mean_material_rgb
from infinigen.assets.utils.object import new_bbox
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_canonical_surfaces
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, int_hash

logger = logging.getLogger(__name__)

# Placeholder is 10 mm thick; the solver seats BACK against the wall, so the
# asset is born ~5 mm proud. Unused film (z=0) must sit just inside the host
# or the card prints; the blister height has to stay *above* that embed or
# only wrinkly peaks remain visible (the previous look).
PLACEHOLDER_HALF_T = 0.005
SOLVER_FLUSH_MARGIN = 0.001
FILM_EMBED = 0.00045


def _host_embed() -> float:
    return PLACEHOLDER_HALF_T + SOLVER_FLUSH_MARGIN + FILM_EMBED


def _smax(a: float, b: float, k: float) -> float:
    if k <= 1e-8:
        return a if a > b else b
    d = abs(a - b)
    h = max(k - d, 0.0) / k
    return (a if a > b else b) + 0.25 * h * h * k


def _blister_cap(x: float, y: float, spec: dict) -> float:
    dx = x - spec["x"]
    dy = y - spec["y"]
    lx = (spec["c"] * dx + spec["s"] * dy) / spec["rx"]
    ly = (-spec["s"] * dx + spec["c"] * dy) / spec["ry"]
    r2 = lx * lx + ly * ly
    if r2 >= 1.0:
        return 0.0
    # Sphere-cap with a slightly flattened crown: stretched paint, not a marble.
    z = math.sqrt(max(0.0, 1.0 - r2))
    z = 0.62 * z + 0.38 * (1.0 - r2)
    return spec["h"] * z


def _sample_cluster(rng, width: float, height: float, style: str) -> list[dict]:
    """Place ellipsoidal blisters in a cluster, sizes in metres."""
    m = 0.16 * min(width, height)
    xs, ys = width * 0.5 - m, height * 0.5 - m
    cx = float(rng.uniform(-xs * 0.45, xs * 0.45))
    cy = float(rng.uniform(-ys * 0.45, ys * 0.45))

    if style == "sparse":
        n_large, n_mid, n_micro = (1, rng.integers(2, 5), rng.integers(6, 14))
        spread = 0.55
    elif style == "dense":
        n_large, n_mid, n_micro = (
            rng.integers(3, 6),
            rng.integers(8, 16),
            rng.integers(22, 40),
        )
        spread = 1.05
    else:
        n_large, n_mid, n_micro = (
            rng.integers(2, 4),
            rng.integers(5, 11),
            rng.integers(12, 26),
        )
        spread = 0.80

    specs = []
    anchors = []

    def _add(kind, r_lo, r_hi, h_lo, h_hi, ox, oy):
        rx = float(rng.uniform(r_lo, r_hi))
        aspect = float(rng.uniform(0.62, 1.0))
        if rng.random() < 0.45:
            aspect = float(rng.uniform(0.45, 0.75))
        ry = max(r_lo * 0.7, rx * aspect)
        ang = float(rng.uniform(0.0, math.pi))
        specs.append(
            {
                "x": ox,
                "y": oy,
                "rx": rx,
                "ry": ry,
                "h": float(rng.uniform(h_lo, h_hi)),
                "c": math.cos(ang),
                "s": math.sin(ang),
                "kind": kind,
            }
        )

    for _ in range(int(n_large)):
        ox = cx + float(rng.uniform(-0.04, 0.04)) * spread
        oy = cy + float(rng.uniform(-0.04, 0.04)) * spread
        _add("large", 0.007, 0.018, 0.0022, 0.0048, ox, oy)
        anchors.append((ox, oy, specs[-1]["rx"]))

    if not anchors:
        anchors.append((cx, cy, 0.02))

    for _ in range(int(n_mid)):
        ax, ay, ar = anchors[int(rng.integers(0, len(anchors)))]
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        rad = float(rng.uniform(0.0, ar * 1.15 * spread))
        _add(
            "mid",
            0.0030,
            0.0085,
            0.0012,
            0.0028,
            ax + math.cos(ang) * rad,
            ay + math.sin(ang) * rad,
        )

    for _ in range(int(n_micro)):
        ax, ay, ar = anchors[int(rng.integers(0, len(anchors)))]
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        rad = float(rng.uniform(0.0, ar * 1.45 * spread))
        _add(
            "micro",
            0.0008,
            0.0026,
            0.00045,
            0.0012,
            ax + math.cos(ang) * rad,
            ay + math.sin(ang) * rad,
        )

    # Keep every blister inside the plane with a small edge dead-zone.
    clipped = []
    for s in specs:
        if abs(s["x"]) + s["rx"] > xs or abs(s["y"]) + s["ry"] > ys:
            continue
        clipped.append(s)
    return clipped or specs[:1]


def _displace(x: float, y: float, specs: list[dict], warp: float) -> float:
    if warp > 1e-6:
        x = x + warp * float(noise.noise(Vector((x * 9.0, y * 9.0, 0.4))))
        y = y + warp * float(noise.noise(Vector((y * 9.0, x * 9.0, 1.7))))
    z = 0.0
    k = 0.0012
    for spec in specs:
        z = _smax(z, _blister_cap(x, y, spec), k)
    return z


def create_bubble_film_material(
    name: str, seed: int, paint_col=(0.82, 0.81, 0.78, 1.0)
) -> bpy.types.Material:
    """Eggshell paint film: same colour as the wall, world-space orange-peel."""
    with FixedSeed(seed):
        peel_scale = float(np.random.uniform(55.0, 95.0))
        grain_scale = float(np.random.uniform(140.0, 260.0))
        roller_scale = float(np.random.uniform(8.0, 22.0))
        bump_dist = float(np.random.uniform(0.0016, 0.0034))
        mottle_amt = float(np.random.uniform(0.10, 0.22))
        rough = float(np.random.uniform(0.52, 0.70))

    lo_s = 0.90
    paint_lo = (
        paint_col[0] * lo_s,
        paint_col[1] * lo_s * 0.995,
        paint_col[2] * lo_s * 0.98,
        1.0,
    )

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

    def math_op(op, a, b=None, clamp=False):
        m = node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, operand in enumerate((a, b)):
            if operand is None:
                continue
            if hasattr(operand, "is_linked"):
                links.new(operand, m.inputs[i])
            else:
                m.inputs[i].default_value = operand
        return m.outputs["Value"]

    attr = node("ShaderNodeAttribute")
    attr.attribute_name = "blister"
    height = attr.outputs["Fac"]

    geom = node("ShaderNodeNewGeometry")
    world = geom.outputs["Position"]
    peel = node("ShaderNodeTexNoise", noise_dimensions="3D", noise_type="FBM")
    peel.inputs["Scale"].default_value = peel_scale
    peel.inputs["Detail"].default_value = 10.0
    peel.inputs["Roughness"].default_value = 0.58
    links.new(world, peel.inputs["Vector"])
    grain = node("ShaderNodeTexNoise", noise_dimensions="3D", noise_type="FBM")
    grain.inputs["Scale"].default_value = grain_scale
    grain.inputs["Detail"].default_value = 8.0
    grain.inputs["Roughness"].default_value = 0.72
    links.new(world, grain.inputs["Vector"])
    roller_map = node("ShaderNodeMapping", vector_type="POINT")
    roller_map.inputs["Scale"].default_value = (0.18, 0.18, 1.0)
    links.new(world, roller_map.inputs["Vector"])
    roller = node("ShaderNodeTexNoise", noise_dimensions="3D", noise_type="FBM")
    roller.inputs["Scale"].default_value = roller_scale
    roller.inputs["Detail"].default_value = 5.0
    links.new(roller_map.outputs["Vector"], roller.inputs["Vector"])
    film_h = math_op(
        "ADD",
        math_op("ADD", math_op("MULTIPLY", peel.outputs["Fac"], 0.55), math_op("MULTIPLY", roller.outputs["Fac"], 0.28)),
        math_op("MULTIPLY", grain.outputs["Fac"], 0.17),
    )

    tone = node("ShaderNodeMixRGB", blend_type="MIX")
    tone.name = "BubblePaintHi"
    tone.inputs["Color1"].default_value = paint_lo
    tone.inputs["Color2"].default_value = paint_col
    links.new(film_h, tone.inputs["Fac"])

    mottle = node("ShaderNodeMixRGB", blend_type="MULTIPLY", use_clamp=True)
    mottle.inputs["Color2"].default_value = tuple(
        min(1.0, c * 0.84) for c in paint_col[:3]
    ) + (1.0,)
    links.new(tone.outputs["Color"], mottle.inputs["Color1"])
    links.new(math_op("MULTIPLY", film_h, mottle_amt, clamp=True), mottle.inputs["Fac"])

    # Stretched film on the crown: slightly smoother bump, same colour.
    bump_h = math_op(
        "MULTIPLY",
        film_h,
        math_op("SUBTRACT", 1.0, math_op("MULTIPLY", height, 0.38)),
    )
    bump = node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.95
    bump.inputs["Distance"].default_value = bump_dist
    links.new(bump_h, bump.inputs["Height"])

    rough_v = math_op(
        "ADD",
        rough,
        math_op("MULTIPLY", math_op("SUBTRACT", 0.5, height), 0.10),
    )
    bsdf = node("ShaderNodeBsdfPrincipled")
    links.new(mottle.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(rough_v, bsdf.inputs["Roughness"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    spec_key = (
        "Specular IOR Level" if "Specular IOR Level" in bsdf.inputs else "Specular"
    )
    if spec_key in bsdf.inputs:
        bsdf.inputs[spec_key].default_value = 0.42

    out = node("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if hasattr(mat, "blend_method"):
        mat.blend_method = "OPAQUE"
    mat.use_backface_culling = True
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "OPAQUE"
    return mat


def _set_bubble_paint_color(mat: bpy.types.Material, rgb) -> None:
    if mat is None or not mat.use_nodes:
        return
    r, g, b = (float(c) for c in rgb[:3])
    hi = (r, g, b, 1.0)
    lo = (r * 0.90, g * 0.90 * 0.995, b * 0.90 * 0.98, 1.0)
    node = mat.node_tree.nodes.get("BubblePaintHi")
    if node is not None:
        node.inputs["Color1"].default_value = lo
        node.inputs["Color2"].default_value = hi


def create_blister_plane(
    width: float,
    height: float,
    seed: int,
) -> bpy.types.Object:
    rng = np.random.default_rng(seed)
    with FixedSeed(seed):
        style = str(np.random.choice(["sparse", "cluster", "dense"], p=[0.22, 0.50, 0.28]))
        warp = float(np.random.uniform(0.0010, 0.0024))

    specs = _sample_cluster(rng, width, height, style)
    cuts = int(np.clip(round(max(width, height) / 0.0024), 90, 320))

    bm = bmesh.new()
    n = cuts
    verts_f = []
    verts_b = []
    heights = []
    hx, hy = width * 0.5, height * 0.5
    z_back = -0.0008

    for i in range(n + 1):
        row_f, row_b, row_h = [], [], []
        y = -hy + (i / n) * height
        for j in range(n + 1):
            x = -hx + (j / n) * width
            z = _displace(x, y, specs, warp)
            row_f.append(bm.verts.new((x, y, z)))
            row_b.append(bm.verts.new((x, y, z_back)))
            row_h.append(z)
        verts_f.append(row_f)
        verts_b.append(row_b)
        heights.append(row_h)

    z_max = max((h for row in heights for h in row), default=0.0) or 1.0

    for i in range(n):
        for j in range(n):
            f = bm.faces.new(
                (
                    verts_f[i][j],
                    verts_f[i][j + 1],
                    verts_f[i + 1][j + 1],
                    verts_f[i + 1][j],
                )
            )
            zq = max(
                heights[i][j],
                heights[i][j + 1],
                heights[i + 1][j],
                heights[i + 1][j + 1],
            )
            f.material_index = 1 if zq > 0.00012 else 0
            b = bm.faces.new(
                (
                    verts_b[i][j],
                    verts_b[i + 1][j],
                    verts_b[i + 1][j + 1],
                    verts_b[i][j + 1],
                )
            )
            b.material_index = 0

    for i in range(n):
        for side in (
            (verts_f[i][0], verts_b[i][0], verts_b[i + 1][0], verts_f[i + 1][0]),
            (verts_f[i][n], verts_f[i + 1][n], verts_b[i + 1][n], verts_b[i][n]),
            (verts_f[0][i], verts_f[0][i + 1], verts_b[0][i + 1], verts_b[0][i]),
            (verts_f[n][i], verts_b[n][i], verts_b[n][i + 1], verts_f[n][i + 1]),
        ):
            sf = bm.faces.new(side)
            sf.material_index = 0

    mesh = bpy.data.meshes.new("WallBubbleMesh")
    bm.to_mesh(mesh)
    bm.free()

    attr = mesh.attributes.new(name="blister", type="FLOAT", domain="POINT")
    # Front verts were created first in each (i,j) pair: front, back, front, back...
    # Creation order: for i, for j: front then back. So even indices are front.
    nvert = (n + 1) * (n + 1)
    vals = [0.0] * (nvert * 2)
    k = 0
    for i in range(n + 1):
        for j in range(n + 1):
            vals[k] = heights[i][j] / z_max
            vals[k + 1] = 0.0
            k += 2
    attr.data.foreach_set("value", vals)

    obj = bpy.data.objects.new("WallBubblePlane", mesh)
    bpy.context.collection.objects.link(obj)

    clear = bpy.data.materials.new(name="WallBubbleClear")
    clear.use_nodes = True
    nt = clear.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if hasattr(clear, "blend_method"):
        clear.blend_method = "CLIP"
    if hasattr(clear, "shadow_method"):
        clear.shadow_method = "NONE"

    film = create_bubble_film_material(f"BubbleMaterial_{id(obj)}", seed)
    obj.data.materials.append(clear)
    obj.data.materials.append(film)
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


class WallBubblePlaneFactory(AssetFactory):
    """Wall paint blisters: clustered spherical-cap lifts of the paint film."""

    def __init__(self, factory_seed, coarse: bool = False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.45, 1.10)

    def create_placeholder(self, **kwargs):
        ph = new_bbox(-0.005, 0.005, -0.5, 0.5, -0.5, 0.5)
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        geom_seed = int_hash((self.factory_seed, kwargs.get("i", 0), "geom"))
        with FixedSeed(geom_seed):
            width = float(np.random.uniform(0.55, 1.0) * self.plane_size)
            height = float(np.random.uniform(0.55, 1.0) * self.plane_size)

        plane = create_blister_plane(width, height, geom_seed)
        plane.name = f"WallBubblePlane_{geom_seed}"
        plane.rotation_euler = (0.0, np.pi / 2, 0.0)
        butil.apply_transform(plane, loc=False, rot=True, scale=True)
        plane["syndefect_surface"] = "wall"
        plane["syndefect_kind"] = "paint_bubble"
        # Domes need to cast a short crescent shadow; unused film is transparent.
        plane.visible_shadow = True
        plane.visible_diffuse = True
        return plane

    def finalize_assets(
        self, assets, state=None, wall_by_name=None, update_embed_transform=True
    ):
        if wall_by_name is None:
            wall_by_name = {
                w.name: w
                for w in bpy.data.objects
                if w.name.endswith(".wall") and w.type == "MESH"
            }

        embed = _host_embed()
        for obj in assets:
            if obj.type != "MESH" or not obj.data.polygons:
                continue
            try:
                wall_mat = _host_material(obj, state, wall_by_name, ".wall")
                rgb = _mean_material_rgb(wall_mat) if wall_mat else None
                _tint_bubble_film(obj, rgb)
                if update_embed_transform:
                    into = obj.matrix_world.to_3x3() @ Vector((-1.0, 0.0, 0.0))
                    if into.length > 1e-8:
                        obj.location += into.normalized() * embed
            except Exception as e:
                logger.warning("Failed to embed wall-bubble plane %s: %s", obj.name, e)


def _host_material(bubble_obj, state, host_by_name, suffix: str):
    """Paint colour from the room host (``.wall`` / ``.ceiling``) this bubble sits on."""
    from infinigen.core import tags as t

    if state is not None and host_by_name:
        for os in state.objs.values():
            if os.obj is not bubble_obj:
                continue
            for rel in os.relations:
                room_name = rel.target_name
                if (
                    room_name in state.objs
                    and t.Semantics.Room in state.objs[room_name].tags
                ):
                    host_name = room_name.split(".")[0] + suffix
                    if host_name in host_by_name:
                        host = host_by_name[host_name]
                        for mat in host.data.materials:
                            if mat is not None:
                                return mat
            break
    nearest, best = None, None
    c = bubble_obj.matrix_world.translation
    for host in host_by_name.values():
        d2 = (c - host.matrix_world.translation).length_squared
        if best is None or d2 < best:
            best, nearest = d2, host
    if nearest is not None:
        for mat in nearest.data.materials:
            if mat is not None:
                return mat
    for host in host_by_name.values():
        for mat in host.data.materials:
            if mat is not None:
                return mat
    return None


def _tint_bubble_film(obj, rgb) -> None:
    if rgb is None:
        rgb = (0.82, 0.81, 0.78)
    if len(obj.data.materials) > 1 and obj.data.materials[1] is not None:
        film = obj.data.materials[1]
        if not (film.name or "").startswith("BubbleMaterial"):
            film.name = f"BubbleMaterial_{int_hash(obj.name)}"
        _set_bubble_paint_color(film, rgb)


class CeilingBubbleFactory(AssetFactory):
    """Same blister film as walls, hung flush to the ceiling (Top vs ceiling).

    Short name on purpose: Blender object names cap at 63 chars.
    """

    def __init__(self, factory_seed, coarse: bool = False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.45, 1.10)

    def create_placeholder(self, **kwargs):
        ph = new_bbox(
            -0.5, 0.5, -0.5, 0.5, -PLACEHOLDER_HALF_T, PLACEHOLDER_HALF_T
        )
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        geom_seed = int_hash((self.factory_seed, kwargs.get("i", 0), "geom"))
        with FixedSeed(geom_seed):
            width = float(np.random.uniform(0.55, 1.0) * self.plane_size)
            height = float(np.random.uniform(0.55, 1.0) * self.plane_size)

        plane = create_blister_plane(width, height, geom_seed)
        # Mesh is born with blisters in +Z. Flip so they hang into the room;
        # local +Z is then the slab, matching the ceiling placeholder.
        plane.rotation_euler = (math.pi, 0.0, 0.0)
        butil.apply_transform(plane, loc=False, rot=True, scale=True)
        plane["syndefect_surface"] = "ceiling"
        plane["syndefect_kind"] = "paint_bubble"
        plane.visible_shadow = True
        plane.visible_diffuse = True
        return plane

    def finalize_assets(
        self, assets, state=None, wall_by_name=None, update_embed_transform=True
    ):
        # ``wall_by_name`` is the host map generate_indoors already passes;
        # for this factory it is ceiling meshes keyed by ``*.ceiling``.
        if wall_by_name is None:
            wall_by_name = {
                c.name: c
                for c in bpy.data.objects
                if c.name.endswith(".ceiling") and c.type == "MESH"
            }

        embed = _host_embed()
        for obj in assets:
            if obj.type != "MESH" or not obj.data.polygons:
                continue
            try:
                host_mat = _host_material(obj, state, wall_by_name, ".ceiling")
                rgb = _mean_material_rgb(host_mat) if host_mat else None
                _tint_bubble_film(obj, rgb)
                into_slab = obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
                if into_slab.length > 1e-8:
                    into_slab.normalize()
                    if update_embed_transform:
                        obj.location += into_slab * embed
                    obj["syndefect_normal"] = tuple(
                        float(c) for c in (-into_slab)
                    )
            except Exception as e:
                logger.warning(
                    "Failed to embed ceiling-bubble plane %s: %s", obj.name, e
                )


def refresh_wall_bubble_materials(wall_objects):
    """Re-tint blister film from current wall materials without moving geometry."""
    bubbles = [
        o
        for o in bpy.data.objects
        if o.type == "MESH" and "WallBubblePlaneFactory" in (o.name or "")
    ]
    if not bubbles:
        return
    wall_by_name = {w.name: w for w in wall_objects if w and w.type == "MESH"}
    if not wall_by_name:
        return
    WallBubblePlaneFactory(factory_seed=0).finalize_assets(
        bubbles, state=None, wall_by_name=wall_by_name, update_embed_transform=False
    )


def refresh_ceiling_bubble_materials(ceiling_objects):
    """Re-tint ceiling blister film after ceiling materials are resampled."""
    bubbles = [
        o
        for o in bpy.data.objects
        if o.type == "MESH" and "CeilingBubbleFactory" in (o.name or "")
    ]
    if not bubbles:
        return
    host_by_name = {c.name: c for c in ceiling_objects if c and c.type == "MESH"}
    if not host_by_name:
        return
    CeilingBubbleFactory(factory_seed=0).finalize_assets(
        bubbles, state=None, wall_by_name=host_by_name, update_embed_transform=False
    )
