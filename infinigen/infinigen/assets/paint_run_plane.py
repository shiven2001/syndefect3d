# Procedural paint-run (drip / sag) plane for Infinigen.
# Same wall-matching mesh workflow as wall bubbles; 1–2 vertical teardrops.

import logging
import math

import bpy
import bmesh
import numpy as np
from mathutils import Vector, noise

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.object import new_bbox
from infinigen.assets.utils.uv import unwrap_normal
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_canonical_surfaces
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.random import weighted_sample

logger = logging.getLogger(__name__)


def _get_fractal_noise(vec: Vector, octaves: int = 3, scale: float = 2.0) -> float:
    val = 0.0
    amp = 1.0
    freq = scale
    for _ in range(octaves):
        val += noise.noise(vec * freq) * amp
        amp *= 0.5
        freq *= 2.0
    return val


def _drip_displacement(
    x: float,
    y: float,
    drips: list,
    noise_scale: float,
    noise_strength: float,
) -> float:
    """Raised tadpole: thin tail along +X (down the wall after Y-rot), bulb at the head."""
    total = 0.0
    for hx, hy, length, r_head, r_tail, hmax in drips:
        tail_x = hx - length
        s = (x - tail_x) / max(length, 1e-6)
        if s < -0.12 or s > 1.25:
            continue
        s_c = max(0.0, min(1.0, s))
        wobble = 0.0
        if s_c < 0.88:
            wobble = (
                _get_fractal_noise(Vector((x * noise_scale, hy, 0.35)), 3, 1.6)
                * noise_strength
                * (1.0 - s_c)
            )
        cy = hy + wobble
        fat = s_c**1.65
        radius = r_tail + (r_head - r_tail) * fat
        if s_c > 0.78:
            bulb = (s_c - 0.78) / 0.22
            radius = radius + r_head * 0.4 * bulb * bulb

        if s >= 1.0:
            dist = math.hypot(x - hx, y - cy)
            radius = r_head * max(0.0, 1.0 - (s - 1.0) * 2.4)
            if radius <= 1e-6:
                continue
        else:
            dist = abs(y - cy)

        if dist >= radius:
            continue
        fall = math.cos((math.pi / 2.0) * (dist / radius))
        h = hmax * (0.18 + 0.82 * fat) * fall
        if s < 0.0:
            h *= max(0.0, 1.0 + s / 0.12)
        total = max(total, h)
    return total


def create_paint_run_plane(
    size: float = 1.0,
    subdivisions: int = 280,
    thickness: float = 0.001,
    num_drips: int = 1,
    edge_margin: float = 0.12,
    seed: int = 0,
) -> bpy.types.Object:
    rng = np.random.default_rng(seed)

    def make_background_material(name):
        mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        node_output = nodes.new(type="ShaderNodeOutputMaterial")
        node_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        node_bsdf.inputs["Alpha"].default_value = 0.0
        links.new(node_bsdf.outputs["BSDF"], node_output.inputs["Surface"])
        return mat

    with FixedSeed(seed):
        wall_mat_gen = weighted_sample(material_assignments.wall)()
        wall_mat = wall_mat_gen(vertical=True)

    bg_mat = make_background_material("PaintRunBackground")

    drips = []
    half = size / 2.0 - edge_margin
    n_want = int(np.clip(num_drips, 1, 2))
    for k in range(n_want):
        length = float(rng.uniform(0.16, 0.38))
        r_head = float(rng.uniform(0.022, 0.045))
        r_tail = float(rng.uniform(0.006, 0.012))
        hmax = float(rng.uniform(0.0035, 0.009))
        hx = float(rng.uniform(-half + length * 0.15, half - r_head * 1.2))
        if n_want == 1:
            hy = float(rng.uniform(-0.08, 0.08))
        else:
            hy = (-0.12 if k == 0 else 0.12) + float(rng.uniform(-0.04, 0.04))
        drips.append((hx, hy, length, r_head, r_tail, hmax))

    noise_scale = float(rng.uniform(2.2, 3.8))
    noise_strength = float(rng.uniform(0.012, 0.028))

    bm = bmesh.new()
    n = subdivisions
    verts_front, verts_back = [], []
    plane_half = size / 2.0

    for i in range(n + 1):
        row_f, row_b = [], []
        for j in range(n + 1):
            x = -plane_half + (j / n) * size
            y = -plane_half + (i / n) * size
            z_f = _drip_displacement(x, y, drips, noise_scale, noise_strength)
            row_f.append(bm.verts.new((x, y, z_f)))
            row_b.append(bm.verts.new((x, y, -thickness)))
        verts_front.append(row_f)
        verts_back.append(row_b)

    bm.faces.ensure_lookup_table()
    for i in range(n):
        for j in range(n):
            f_face = bm.faces.new(
                (
                    verts_front[i][j],
                    verts_front[i][j + 1],
                    verts_front[i + 1][j + 1],
                    verts_front[i + 1][j],
                )
            )
            f_face.material_index = (
                1 if any(v.co.z > 0.00015 for v in f_face.verts) else 0
            )
            b_face = bm.faces.new(
                (
                    verts_back[i][j],
                    verts_back[i + 1][j],
                    verts_back[i + 1][j + 1],
                    verts_back[i][j + 1],
                )
            )
            b_face.material_index = 0

    for i in range(n):
        sides = [
            (
                verts_front[i][0],
                verts_back[i][0],
                verts_back[i + 1][0],
                verts_front[i + 1][0],
            ),
            (
                verts_front[i][n],
                verts_front[i + 1][n],
                verts_back[i + 1][n],
                verts_back[i][n],
            ),
            (
                verts_front[0][i],
                verts_front[0][i + 1],
                verts_back[0][i + 1],
                verts_back[0][i],
            ),
            (
                verts_front[n][i],
                verts_back[n][i],
                verts_back[n][i + 1],
                verts_front[n][i + 1],
            ),
        ]
        for side in sides:
            s_face = bm.faces.new(side)
            s_face.material_index = 0

    mesh = bpy.data.meshes.new("PaintRunMesh")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("PaintRunPlane", mesh)
    bpy.context.collection.objects.link(obj)
    unwrap_normal(obj)
    obj.data.materials.append(bg_mat)
    obj.data.materials.append(wall_mat)
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


def _modify_wall_mat_for_run(mat: bpy.types.Material) -> bpy.types.Material:
    """Same paint as the wall, slightly glossier so the raised drip catches light."""
    mat = mat.copy()
    if not mat.use_nodes or not mat.node_tree:
        return mat
    nt = mat.node_tree
    links = nt.links
    for node in nt.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        rough = node.inputs.get("Roughness")
        if rough is None:
            continue
        incoming = [lk for lk in links if lk.to_socket == rough]
        if incoming:
            mul = nt.nodes.new(type="ShaderNodeMath")
            mul.operation = "MULTIPLY"
            mul.inputs[1].default_value = 0.82
            src = incoming[0].from_socket
            links.remove(incoming[0])
            links.new(src, mul.inputs[0])
            links.new(mul.outputs["Value"], rough)
        else:
            rough.default_value = min(float(rough.default_value), 0.48)
    return mat


class PaintRunPlaneFactory(AssetFactory):
    """1–2 small vertical paint drips, same placement / wall-paint match as bubbles."""

    def __init__(self, factory_seed, coarse: bool = False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.22, 0.42)

    def create_placeholder(self, **kwargs):
        ph = new_bbox(-0.005, 0.005, -0.5, 0.5, -0.5, 0.5)
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        geom_seed = int_hash((self.factory_seed, kwargs.get("i", 0), "geom"))
        with FixedSeed(geom_seed):
            scale_z_val = np.random.uniform(0.7, 1.0) * self.plane_size / 2
            scale_y_val = np.random.uniform(0.55, 0.9) * self.plane_size / 2
            num_drips = 1 if np.random.rand() < 0.62 else 2

        plane = create_paint_run_plane(
            size=1.0,
            subdivisions=280,
            thickness=0.001,
            num_drips=num_drips,
            edge_margin=0.12,
            seed=geom_seed,
        )
        plane.name = f"PaintRunPlane_{geom_seed}"
        plane.scale = (scale_z_val, scale_y_val, 1.0)
        plane.rotation_euler = (0.0, np.pi / 2, 0.0)
        butil.apply_transform(plane, loc=False, rot=True, scale=True)
        plane.visible_shadow = True
        plane.visible_diffuse = True
        return plane

    def finalize_assets(
        self, assets, state=None, wall_by_name=None, update_embed_transform=True
    ):
        EMBED_OFFSET = -0.02

        from infinigen.core import tags as t
        from infinigen.core.tagging import tagged_face_mask

        if wall_by_name is None:
            wall_by_name = {
                w.name: w
                for w in bpy.data.objects
                if w.name.endswith(".wall") and w.type == "MESH"
            }

        def _get_wall_mat(run_obj):
            if state is not None and wall_by_name:
                for os in state.objs.values():
                    if os.obj is not run_obj:
                        continue
                    for rel in os.relations:
                        room_name = rel.target_name
                        if (
                            room_name in state.objs
                            and t.Semantics.Room in state.objs[room_name].tags
                        ):
                            wall_name = room_name.split(".")[0] + ".wall"
                            if wall_name in wall_by_name:
                                wall_obj = wall_by_name[wall_name]
                                for mat in wall_obj.data.materials:
                                    if mat is not None:
                                        return mat
                    break
            nearest_wall, best_d2 = None, None
            center = run_obj.matrix_world.translation
            for w in wall_by_name.values():
                d2 = (center - w.matrix_world.translation).length_squared
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    nearest_wall = w
            if nearest_wall is not None:
                for mat in nearest_wall.data.materials:
                    if mat is not None:
                        return mat
            for w in wall_by_name.values():
                for mat in w.data.materials:
                    if mat is not None:
                        return mat
            return None

        for obj in assets:
            if obj.type != "MESH" or not obj.data.polygons:
                continue
            try:
                wall_mat = _get_wall_mat(obj)
                if wall_mat is not None and len(obj.data.materials) > 1:
                    run_mat = _modify_wall_mat_for_run(wall_mat)
                    run_mat.name = f"PaintRunMaterial_{int_hash(obj.name)}"
                    obj.data.materials[1] = run_mat

                if update_embed_transform:
                    back_mask = tagged_face_mask(obj, {t.Subpart.Back})
                    if back_mask.any():
                        back_faces = [i for i, tag in enumerate(back_mask) if tag]
                        if not back_faces:
                            continue
                        back_poly = obj.data.polygons[
                            max(back_faces, key=lambda idx: obj.data.polygons[idx].area)
                        ]
                    else:
                        back_poly = max(
                            obj.data.polygons,
                            key=lambda p: -p.normal.y if p.normal.y < 0 else -1e6,
                        )
                    wall_normal = Vector(
                        butil.global_polygon_normal(obj, back_poly)
                    ).normalized()
                    obj.location += wall_normal * EMBED_OFFSET
            except Exception as e:
                logger.warning("Failed to embed paint-run plane %s: %s", obj.name, e)


def refresh_paint_run_materials(wall_objects):
    runs = [
        o
        for o in bpy.data.objects
        if o.type == "MESH" and o.name.startswith("PaintRunPlane")
    ]
    if not runs:
        return
    wall_by_name = {w.name: w for w in wall_objects if w and w.type == "MESH"}
    if not wall_by_name:
        return
    PaintRunPlaneFactory(factory_seed=0).finalize_assets(
        runs, state=None, wall_by_name=wall_by_name, update_embed_transform=False
    )
