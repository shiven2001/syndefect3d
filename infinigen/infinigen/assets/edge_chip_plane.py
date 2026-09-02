# Convex-arris paint chips (door reveals, pillars, beam soffits).
#
# These live on edges, not faces, so they cannot go through the furniture
# solver (flush_wall_defect seats a plane on a tagged wall face). Placement
# is a post-solve pass over convex mesh edges after walls / pillars / beams
# exist. Intact paint is transparent; only the chips carry CornerChipMaterial
# so material-index export can label them.

import logging
import math
from dataclasses import dataclass

import bpy
import bmesh
import gin
import numpy as np
from mathutils import Matrix, Vector

from infinigen.assets.paint_peel_plane import scene_peel_substrate
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, int_hash

logger = logging.getLogger(__name__)

MATERIAL_PREFIX = "CornerChipMaterial"
# The strips are alpha-cutout decals with no displacement output, so any
# part of them left inside the solid is simply hidden by it. They were
# embedded twice - once per wing along the face normal, then again along the
# bisector - which buried them ~0.77 mm deep and made the chipping invisible.
# Decals have to sit a hair PROUD of the faces instead. Shadow, diffuse and
# glossy visibility are already off (see _disable_chip_lighting), so a proud
# strip cannot print its own footprint onto the plaster.
PROUD_M = 0.0003
WING_WIDTH = 0.014


def _generation_seed():
    try:
        return int(gin.query_parameter("OVERALL_SEED"))
    except Exception:
        return 0


def _disable_chip_lighting(obj: bpy.types.Object) -> None:
    obj.visible_shadow = False
    obj.visible_diffuse = False
    if hasattr(obj, "visible_glossy"):
        obj.visible_glossy = False
    if hasattr(obj, "visible_transmission"):
        obj.visible_transmission = False
    if hasattr(obj, "visible_volume_scatter"):
        obj.visible_volume_scatter = False


def _room_is_kept(name, keep_rooms):
    if keep_rooms is None:
        return True
    return any(k.split(".")[0] in name for k in keep_rooms)


def _is_tiled(obj):
    if obj is None:
        return False
    if obj.get("surface_finish") == "tile":
        return True
    return False


def _face_tangent(n_self: Vector, n_other: Vector, edge: Vector) -> Vector | None:
    """In-face direction from the arris onto this face, toward the solid."""
    t = n_self.cross(edge)
    if t.length < 1e-6:
        return None
    t.normalize()
    if t.dot(n_other) > 0:
        t = -t
    return t


def _bisector_faces_open_air(arris: "_Arris") -> bool:
    """True if the arris is a knockable convex edge (solid behind, air in front)."""
    bis = arris.n1 + arris.n2
    if bis.length < 0.15:
        return False
    bis.normalize()
    mid = 0.5 * (arris.p0 + arris.p1)
    start = mid + bis * 0.025
    mi = arris.host.matrix_world.inverted()
    loc_start = mi @ start
    loc_dir = mi.to_3x3() @ (-bis)
    if loc_dir.length < 1e-6:
        return False
    loc_dir.normalize()
    hit, _, _, _ = arris.host.ray_cast(loc_start, loc_dir, distance=0.22)
    return bool(hit)


def _has_room_clearance(arris: "_Arris", depsgraph, min_clear: float = 0.6) -> bool:
    """True if the space in front of the arris is actually the room.

    `_bisector_faces_open_air` only proves there is solid behind the edge, and
    that is equally true of an arris on the wall's cavity-facing side. Those sit
    in the ~0.2 m gap between the interior wall and the exterior shell, where
    nothing in the room can ever see them - every chip in a test scene landed
    there. Requiring real clearance in front rejects the cavity while leaving
    genuine room-side corners (which have metres of space) untouched.
    """
    bis = arris.n1 + arris.n2
    if bis.length < 0.15:
        return False
    bis.normalize()
    mid = 0.5 * (arris.p0 + arris.p1)
    edge = (arris.p1 - arris.p0)
    if edge.length < 1e-6:
        return False
    edge.normalize()
    # A single ray down the bisector misses trim that sits BESIDE the corner:
    # door casings occluded several corners that passed the straight-ahead
    # test. Sweep a fan about the edge and require the whole approach cone to
    # be clear, so the corner is actually approachable by a camera.
    for deg in (-38.0, -20.0, 0.0, 20.0, 38.0):
        d = bis.copy()
        d.rotate(Matrix.Rotation(math.radians(deg), 3, edge))
        hit, _, _, _, _, _ = bpy.context.scene.ray_cast(
            depsgraph, mid + d * 0.03, d, distance=min_clear
        )
        if hit:
            return False
    return True


def _axes_matrix(x: Vector, y: Vector, z: Vector, origin: Vector) -> Matrix:
    m = Matrix.Identity(4)
    m[0][0], m[1][0], m[2][0] = x
    m[0][1], m[1][1], m[2][1] = y
    m[0][2], m[1][2], m[2][2] = z
    m.translation = origin
    return m


@dataclass
class _Arris:
    p0: Vector
    p1: Vector
    n1: Vector
    n2: Vector
    kind: str
    host: object


def _iter_convex_edges(obj, *, vertical: bool, min_len: float, max_len: float):
    if obj is None or obj.type != "MESH" or obj.data is None or not obj.data.polygons:
        return
    deps = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(deps)
    mesh = eval_obj.to_mesh()
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        mw = obj.matrix_world
        rot = mw.to_3x3()
        bbox = [mw @ Vector(c) for c in obj.bound_box]
        zmin = min(c.z for c in bbox)
        zmax = max(c.z for c in bbox)
        for e in bm.edges:
            if len(e.link_faces) != 2:
                continue
            if not e.is_convex:
                continue
            v0 = mw @ e.verts[0].co
            v1 = mw @ e.verts[1].co
            d = v1 - v0
            length = d.length
            if length < min_len or length > max_len:
                continue
            direction = d.normalized()
            if vertical:
                if abs(direction.z) < 0.85:
                    continue
            else:
                if abs(direction.z) > 0.25:
                    continue
                mid_z = 0.5 * (v0.z + v1.z)
                # Bottom soffit / lower arris, not the slab junction.
                if mid_z > zmin + 0.10 and mid_z < zmax - 0.04:
                    if mid_z > (zmin + zmax) * 0.5:
                        continue
            n1 = rot @ Vector(e.link_faces[0].normal)
            n2 = rot @ Vector(e.link_faces[1].normal)
            if n1.length < 1e-5 or n2.length < 1e-5:
                continue
            n1.normalize()
            n2.normalize()
            # ~90° plus bevelled 135° chamfers on pillars.
            if abs(n1.dot(n2)) > 0.82:
                continue
            if v0.z <= v1.z:
                p0, p1 = v0, v1
            else:
                p0, p1 = v1, v0
                if vertical:
                    direction = -direction
            kind = "vertical" if vertical else "horizontal"
            arris = _Arris(p0.copy(), p1.copy(), n1.copy(), n2.copy(), kind, obj)
            if not _bisector_faces_open_air(arris):
                continue
            yield arris
    finally:
        bm.free()
        eval_obj.to_mesh_clear()


def _dedup(arrises, radius=0.055):
    kept = []
    for a in sorted(arrises, key=lambda x: -(x.p1 - x.p0).length):
        mid = 0.5 * (a.p0 + a.p1)
        d = (a.p1 - a.p0).normalized()
        clash = False
        for b in kept:
            bmid = 0.5 * (b.p0 + b.p1)
            bd = (b.p1 - b.p0).normalized()
            if (mid - bmid).length < radius and abs(d.dot(bd)) > 0.85:
                clash = True
                break
        if not clash:
            kept.append(a)
    return kept


def _crop_segment(arris: _Arris) -> tuple[Vector, Vector] | None:
    p0, p1 = arris.p0.copy(), arris.p1.copy()
    full = (p1 - p0).length
    if full < 0.16:
        return None
    if arris.kind == "vertical":
        # Keep off the floor slab and ceiling; chips cluster like the refs.
        z0 = p0.z + 0.07
        z1 = p1.z - 0.06
        if z1 - z0 < 0.20:
            return None
        span = float(np.random.uniform(0.22, min(0.70, z1 - z0)))
        lo = float(np.random.uniform(z0, z1 - span))
        t0 = (lo - p0.z) / (p1.z - p0.z + 1e-9)
        t1 = (lo + span - p0.z) / (p1.z - p0.z + 1e-9)
        t0 = float(np.clip(t0, 0.0, 1.0))
        t1 = float(np.clip(t1, 0.0, 1.0))
        if t1 - t0 < 0.12:
            return None
        return p0.lerp(p1, t0), p0.lerp(p1, t1)
    span = float(np.random.uniform(0.22, min(0.80, full * 0.70)))
    t0 = float(np.random.uniform(0.05, max(0.06, 1.0 - span / full - 0.05)))
    t1 = t0 + span / full
    return p0.lerp(p1, t0), p0.lerp(p1, min(t1, 0.95))


def _sample_chip_layout(u_span: float, v_span: float):
    """Sparse jagged chips along the arris, never a continuous bead.

    Sizes are sampled in METRES and converted to UV here. The strip's UV is
    strongly anisotropic - u spans both wings (~30 mm) while v spans the whole
    arris (~0.6 m), roughly 19:1 - so radii specified directly in UV came out
    as hair-thin vertical needles instead of the few-mm chunks a chipped
    corner actually loses.
    """
    specs = []
    su = 1.0 / max(u_span, 1e-6)
    sv = 1.0 / max(v_span, 1e-6)
    n_clusters = int(np.random.randint(2, 5))
    for _ in range(n_clusters):
        cy = float(np.random.uniform(-0.28, 0.28))  # UV fraction along arris
        n_chips = int(np.random.randint(1, 4))
        for i in range(n_chips):
            specs.append(
                {
                    "cx": float(np.random.uniform(-0.0008, 0.0008)) * su,
                    "cy": cy + float(np.random.uniform(-0.018, 0.018)) * sv,
                    "rx": float(
                        np.random.uniform(0.0020, 0.0055)
                        if i == 0
                        else np.random.uniform(0.0012, 0.0032)
                    )
                    * su,
                    "ry": float(np.random.uniform(0.0030, 0.0120)) * sv,
                    "rot": float(np.random.uniform(-0.45, 0.45)),
                    "mode": "add",
                }
            )
            if np.random.rand() < 0.28:
                specs.append(
                    {
                        "cx": specs[-1]["cx"]
                        + float(np.random.uniform(-0.0006, 0.0006)) * su,
                        "cy": specs[-1]["cy"]
                        + float(np.random.uniform(-0.0020, 0.0020)) * sv,
                        "rx": specs[-1]["rx"] * float(np.random.uniform(0.35, 0.70)),
                        "ry": specs[-1]["ry"] * float(np.random.uniform(0.35, 0.70)),
                        "rot": float(np.random.uniform(-0.5, 0.5)),
                        "mode": "sub",
                    }
                )
    n_specks = int(np.random.randint(1, 4))
    for _ in range(n_specks):
        specs.append(
            {
                "cx": float(np.random.uniform(-0.0006, 0.0006)) * su,
                "cy": float(np.random.uniform(-0.28, 0.28)),
                "rx": float(np.random.uniform(0.0008, 0.0022)) * su,
                "ry": float(np.random.uniform(0.0012, 0.0035)) * sv,
                "rot": float(np.random.uniform(0.0, 6.28)),
                "mode": "add",
            }
        )
    return specs


def create_corner_chip_material(
    name: str, seed: int, u_span: float = 0.03, v_span: float = 0.6
) -> bpy.types.Material:
    """Shallow nicks on a convex arris: peel-substrate plaster, not a dark bead."""
    plaster_lo, plaster_hi, pit_tint, _flake = scene_peel_substrate()
    with FixedSeed(seed):
        specs = _sample_chip_layout(u_span, v_span)
        warp_scale = np.random.uniform(40.0, 90.0)
        warp_amp = np.random.uniform(0.016, 0.036)
        teeth_scale = np.random.uniform(250.0, 600.0)
        teeth_amp = np.random.uniform(0.08, 0.18)
        cell_scale = np.random.uniform(120.0, 300.0)
        cell_amp = np.random.uniform(0.05, 0.12)
        bump_s = np.random.uniform(0.14, 0.32)
        bump_d = np.random.uniform(0.00025, 0.00070)
        grain_scale = np.random.uniform(55.0, 120.0)
        mapping_offset = tuple(np.random.uniform(0, 80, 3))

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

    def noise(vector, scale, detail=6.0, roughness=0.55):
        n = node("ShaderNodeTexNoise", noise_dimensions="3D", noise_type="FBM")
        if hasattr(n, "normalize"):
            n.normalize = True
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = roughness
        links.new(vector, n.inputs["Vector"])
        return n.outputs["Fac"]

    node_output = node("ShaderNodeOutputMaterial")
    uvmap = node("ShaderNodeUVMap")
    uvmap.uv_map = "UVMap"
    tex = node("ShaderNodeTexCoord")
    pattern_map = node("ShaderNodeMapping", vector_type="POINT")
    pattern_map.inputs["Location"].default_value = mapping_offset
    links.new(tex.outputs["Object"], pattern_map.inputs["Vector"])
    pattern = pattern_map.outputs["Vector"]

    sep = node("ShaderNodeSeparateXYZ")
    links.new(uvmap.outputs["UV"], sep.inputs["Vector"])
    u = math("SUBTRACT", sep.outputs["X"], 0.5)
    v = math("SUBTRACT", sep.outputs["Y"], 0.5)
    uv = node("ShaderNodeCombineXYZ")
    links.new(u, uv.inputs["X"])
    links.new(v, uv.inputs["Y"])
    uvv = uv.outputs["Vector"]

    # Metric copy of the UV, in metres, so noise and Voronoi features come out
    # round on the wall instead of smeared along the arris.
    uvm_n = node("ShaderNodeCombineXYZ")
    links.new(math("MULTIPLY", u, u_span), uvm_n.inputs["X"])
    links.new(math("MULTIPLY", v, v_span), uvm_n.inputs["Y"])
    uvm = uvm_n.outputs["Vector"]

    wu = math(
        "MULTIPLY",
        math("SUBTRACT", noise(uvm, warp_scale, 5.0, 0.62), 0.5),
        warp_amp * 2.0,
    )
    wv = math(
        "MULTIPLY",
        math("SUBTRACT", noise(pattern, warp_scale * 1.31, 5.0, 0.58), 0.5),
        warp_amp * 2.0,
    )
    u_w = math("ADD", u, wu)
    v_w = math("ADD", v, wv)
    teeth = math(
        "MULTIPLY",
        math("SUBTRACT", noise(uvm, teeth_scale, 8.0, 0.78), 0.5),
        teeth_amp,
    )
    cells = node(
        "ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="DISTANCE_TO_EDGE"
    )
    cells.inputs["Scale"].default_value = cell_scale
    cells.inputs["Randomness"].default_value = 1.0
    links.new(uvm, cells.inputs["Vector"])
    jag = math(
        "ADD",
        teeth,
        # Voronoi Distance is already in scaled units, so it needs no
        # further rescaling - multiplying by cell_scale drove jag far
        # negative and forced alpha to 1 along the whole arris.
        math("MULTIPLY", math("SUBTRACT", 0.12, cells.outputs["Distance"]), cell_amp),
    )

    def flake_mask(spec):
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
        return ramp(math("ADD", edist, jag), 0.86, 1.08, 1.0, 0.0)

    add_specs = [s for s in specs if s["mode"] == "add"]
    sub_specs = [s for s in specs if s["mode"] == "sub"]
    chips = flake_mask(add_specs[0])
    for spec in add_specs[1:]:
        chips = math("MAXIMUM", chips, flake_mask(spec), clamp=True)
    for spec in sub_specs:
        bite = flake_mask(spec)
        chips = math(
            "MULTIPLY",
            chips,
            math("SUBTRACT", 1.0, math("MULTIPLY", bite, 0.90)),
            clamp=True,
        )

    # Die before the wing tips so chips stay on the arris, not the flat face.
    keep_u = ramp(math("ABSOLUTE", u), 0.16, 0.30, 1.0, 0.0, interp="SMOOTHSTEP")
    keep_v = ramp(math("ABSOLUTE", v), 0.40, 0.49, 1.0, 0.0, interp="SMOOTHSTEP")
    chips = math("MULTIPLY", math("MULTIPLY", chips, keep_u, clamp=True), keep_v, clamp=True)

    n_body = node("ShaderNodeTexNoise", noise_dimensions="3D", noise_type="FBM")
    if hasattr(n_body, "normalize"):
        n_body.normalize = True
    n_body.inputs["Scale"].default_value = 18.0
    n_body.inputs["Detail"].default_value = 12.0
    n_body.inputs["Roughness"].default_value = 0.66
    links.new(tex.outputs["Object"], n_body.inputs["Vector"])

    plaster_ramp = node("ShaderNodeValToRGB")
    plaster_ramp.color_ramp.elements[0].position = 0.18
    plaster_ramp.color_ramp.elements[1].position = 0.82
    plaster_ramp.color_ramp.elements[0].color = plaster_lo
    plaster_ramp.color_ramp.elements[1].color = plaster_hi
    links.new(n_body.outputs["Fac"], plaster_ramp.inputs["Fac"])

    n_grain = node("ShaderNodeTexNoise", noise_dimensions="3D", noise_type="FBM")
    if hasattr(n_grain, "normalize"):
        n_grain.normalize = True
    n_grain.inputs["Scale"].default_value = grain_scale
    n_grain.inputs["Detail"].default_value = 10.0
    n_grain.inputs["Roughness"].default_value = 0.78
    links.new(tex.outputs["Object"], n_grain.inputs["Vector"])

    grain_mix = node("ShaderNodeMixRGB", blend_type="MIX")
    grain_mix.inputs["Fac"].default_value = 0.08
    links.new(plaster_ramp.outputs["Color"], grain_mix.inputs["Color1"])
    links.new(n_grain.outputs["Color"], grain_mix.inputs["Color2"])

    pit_mix = node("ShaderNodeMixRGB", blend_type="MULTIPLY")
    pit_mix.inputs["Fac"].default_value = 0.18
    pit_mix.inputs["Color2"].default_value = pit_tint
    links.new(grain_mix.outputs["Color"], pit_mix.inputs["Color1"])

    bump = node("ShaderNodeBump")
    bump.invert = True
    bump.inputs["Strength"].default_value = bump_s
    bump.inputs["Distance"].default_value = bump_d
    links.new(chips, bump.inputs["Height"])

    bsdf = node("ShaderNodeBsdfPrincipled")
    links.new(pit_mix.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.78
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(chips, bsdf.inputs["Alpha"])
    spec_key = (
        "Specular IOR Level" if "Specular IOR Level" in bsdf.inputs else "Specular"
    )
    if spec_key in bsdf.inputs:
        bsdf.inputs[spec_key].default_value = 0.10
    links.new(bsdf.outputs["BSDF"], node_output.inputs["Surface"])

    mat.blend_method = "CLIP"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"
    if hasattr(mat, "alpha_threshold"):
        mat.alpha_threshold = 0.40
    if hasattr(mat, "use_transparent_shadow"):
        mat.use_transparent_shadow = False
    return mat


def _spawn_v_strip(p0: Vector, p1: Vector, n1: Vector, n2: Vector, seed: int, name: str):
    edge = p1 - p0
    length = edge.length
    if length < 0.12:
        return None
    ez = edge.normalized()
    t1 = _face_tangent(n1, n2, ez)
    t2 = _face_tangent(n2, n1, ez)
    if t1 is None or t2 is None:
        return None
    bis = n1 + n2
    if bis.length < 0.15:
        return None
    bis.normalize()
    ey = ez.cross(bis)
    if ey.length < 1e-5:
        ey = t1.cross(ez)
    ey.normalize()
    bis = ey.cross(ez).normalized()
    mid = 0.5 * (p0 + p1)
    mw = _axes_matrix(bis, ey, ez, mid)
    inv = mw.inverted()

    def to_local(vec: Vector) -> Vector:
        return inv.to_3x3() @ vec

    t1l = to_local(t1)
    t2l = to_local(t2)
    n1l = to_local(n1)
    n2l = to_local(n2)
    width = WING_WIDTH
    n_along = int(np.clip(round(length / 0.014), 16, 72))
    n_across = 3

    bm = bmesh.new()
    uv_lay = bm.loops.layers.uv.new("UVMap")

    def add_wing(t_across: Vector, n_face: Vector, side: float):
        embed = n_face.normalized() * PROUD_M
        verts = []
        for i in range(n_along + 1):
            fv = i / n_along
            along = Vector((0.0, 0.0, (fv - 0.5) * length))
            row = []
            for j in range(n_across + 1):
                fu = j / n_across
                co = along + t_across * (fu * width) + embed
                row.append(bm.verts.new(co))
            verts.append(row)
        for i in range(n_along):
            for j in range(n_across):
                face = bm.faces.new(
                    (verts[i][j], verts[i + 1][j], verts[i + 1][j + 1], verts[i][j + 1])
                )
                for loop in face.loops:
                    co = loop.vert.co
                    dist = (co - Vector((0.0, 0.0, co.z))).length
                    u = 0.5 + side * 0.5 * min(dist / max(width, 1e-6), 1.0)
                    v = 0.5 + co.z / max(length, 1e-6)
                    loop[uv_lay].uv = (float(u), float(v))

    add_wing(t1l, n1l, 1.0)
    add_wing(t2l, n2l, -1.0)
    bm.normal_update()
    for face in bm.faces:
        if face.normal.x < 0:
            face.normal_flip()

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.matrix_world = mw
    mat = create_corner_chip_material(
        f"{MATERIAL_PREFIX}_{id(obj)}",
        seed,
        u_span=2.0 * width,
        v_span=length,
    )
    obj.data.materials.append(mat)
    obj["syndefect_surface"] = "wall"
    obj["syndefect_kind"] = "edge_chip"
    _disable_chip_lighting(obj)
    return obj


def _collect_hosts(walls, beams, pillars, keep_rooms):
    kept_walls = [
        w
        for w in (walls or [])
        if w is not None
        and w.type == "MESH"
        and _room_is_kept(w.name, keep_rooms)
        and not _is_tiled(w)
    ]
    extra = []
    for group in (beams or []), (pillars or []):
        for obj in group:
            if obj is None or obj.type != "MESH":
                continue
            extra.append(obj)
    if keep_rooms is not None and kept_walls:
        corners = []
        for w in kept_walls:
            corners.extend(w.matrix_world @ Vector(c) for c in w.bound_box)
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        pad = 1.2
        x0, x1 = min(xs) - pad, max(xs) + pad
        y0, y1 = min(ys) - pad, max(ys) + pad

        def in_room(o):
            p = o.matrix_world.translation
            return x0 <= p.x <= x1 and y0 <= p.y <= y1

        extra = [o for o in extra if in_room(o)]
    return kept_walls, extra


@gin.configurable
def install_edge_chips(
    walls,
    beams=None,
    pillars=None,
    keep_rooms=None,
    enabled=True,
    n_vertical=(1, 3),
    n_beam=(0, 2),
    seed=None,
):
    """Place jagged paint-chip strips on convex vertical corners and beam soffits."""
    if not enabled:
        logger.info("Skipping edge chips")
        return []
    if seed is None:
        seed = _generation_seed()
    walls, solids = _collect_hosts(walls, beams, pillars, keep_rooms)
    hosts_vertical = list(walls) + list(solids)
    arrises = []
    for obj in hosts_vertical:
        arrises.extend(
            _iter_convex_edges(obj, vertical=True, min_len=0.22, max_len=4.5)
        )
    beam_arrises = []
    for obj in solids:
        beam_arrises.extend(
            _iter_convex_edges(obj, vertical=False, min_len=0.30, max_len=8.0)
        )
    depsgraph = bpy.context.evaluated_depsgraph_get()
    n_raw, n_beam_raw = len(arrises), len(beam_arrises)
    arrises = [a for a in arrises if _has_room_clearance(a, depsgraph)]
    beam_arrises = [a for a in beam_arrises if _has_room_clearance(a, depsgraph)]
    logger.info(
        "Edge chips: %s/%s vertical and %s/%s beam arrises face the room",
        len(arrises),
        n_raw,
        len(beam_arrises),
        n_beam_raw,
    )
    arrises = _dedup(arrises)
    beam_arrises = _dedup(beam_arrises)
    spawned = []

    with FixedSeed(int_hash((seed, "edge_chips"))):
        n_v = int(np.random.randint(int(n_vertical[0]), int(n_vertical[1]) + 1))
        n_h = int(np.random.randint(int(n_beam[0]), int(n_beam[1]) + 1))
        rng = np.random.default_rng(int_hash((seed, "edge_chips_pick")))

        def pick(cands, k):
            """Sample without replacement, weighted toward longer arrises.

            A uniform shuffle kept landing on the short stubs beside door
            openings; the long runs are the wall returns the reference photos
            actually show chipping on.
            """
            cands = list(cands)
            if not cands or k <= 0:
                return []
            w = np.array([(a.p1 - a.p0).length for a in cands], dtype=float)
            w = np.clip(w, 1e-3, None) ** 2
            k = min(k, len(cands))
            idx = rng.choice(len(cands), size=k, replace=False, p=w / w.sum())
            return [cands[i] for i in idx]

        chosen = pick(arrises, n_v) + pick(beam_arrises, n_h)

        col = butil.get_collection("unique_assets:edge_chips")
        spawned = []
        for i, arris in enumerate(chosen):
            cropped = _crop_segment(arris)
            if cropped is None:
                continue
            a0, a1 = cropped
            obj = _spawn_v_strip(
                a0,
                a1,
                arris.n1,
                arris.n2,
                int_hash((seed, "edge_chip", i, arris.host.name)),
                f"EdgeChip_{i:03d}",
            )
            if obj is None:
                continue
            butil.put_in_collection(obj, col)
            spawned.append(obj)

    if spawned:
        bpy.context.view_layer.update()
    logger.info(
        "Edge chips: %s strips (%s vertical candidates, %s beam candidates)",
        len(spawned),
        len(arrises),
        len(beam_arrises),
    )
    return spawned
