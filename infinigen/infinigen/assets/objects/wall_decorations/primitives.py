# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.

"""Small mesh helpers for wall-mounted procedural fixtures."""

import bpy
from mathutils import Vector

from infinigen.core.util import blender as butil


def box(size, location=(0.0, 0.0, 0.0), name="box"):
    """Axis-aligned box of world size ``(sx, sy, sz)`` centered at ``location``."""
    sx, sy, sz = size
    obj = butil.spawn_cube(size=1, location=location, scale=(sx, sy, sz), name=name)
    butil.apply_transform(obj, loc=True)
    return obj


def cylinder(radius, depth, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), name="cyl"):
    obj = butil.spawn_cylinder(radius=radius, depth=depth, location=location, name=name)
    obj.rotation_euler = rotation
    butil.apply_transform(obj, loc=True)
    return obj


def shade_smooth(obj):
    mesh = obj.data
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = 0.7
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


def solid_material(name, color, roughness=0.35, metallic=0.0):
    """Predictable Principled material; ``color`` is RGB or RGBA in 0–1."""
    if len(color) == 3:
        color = (*color, 1.0)
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = next(
            n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"
        )
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = metallic
    return mat


def assign(obj, mat):
    obj.active_material = mat
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj


def shift_min_x_to_zero(obj):
    """Put the back face on x=0 so flush_wall can snap against the wall."""
    xs = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_x = min(v.x for v in xs)
    obj.location.x -= min_x
    butil.apply_transform(obj, loc=True)
    return obj
