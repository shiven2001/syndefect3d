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


def plastic_material(name, color, roughness=0.35, sheen_scale=None, variation=0.12):
    """Moulded ABS fascia: roughness breaks up over the surface instead of sitting
    at one value, which is what makes a flat `solid_material` panel read as CAD."""
    if len(color) == 3:
        color = (*color, 1.0)
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = color

    # Fine noise -> roughness, so highlights crawl the way they do on real moulded
    # plastic rather than forming one clean specular sheet.
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = (
        sheen_scale if sheen_scale is not None else 180.0
    )
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.6
    nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = 0.0
    ramp.inputs["From Max"].default_value = 1.0
    ramp.inputs["To Min"].default_value = max(0.04, roughness - variation)
    ramp.inputs["To Max"].default_value = min(1.0, roughness + variation)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
    nt.links.new(ramp.outputs["Result"], bsdf.inputs["Roughness"])

    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.42
    return mat


def rounded_box(size, location=(0.0, 0.0, 0.0), radius=0.004, segments=3, name="box"):
    """Box with eased arrises - moulded parts have no truly sharp edges."""
    obj = box(size, location=location, name=name)
    butil.modify_mesh(obj, "BEVEL", width=radius, segments=segments, limit_method="ANGLE")
    shade_smooth(obj)
    return obj


def glass_material(name, roughness=0.035, ior=1.48):
    """Clear architectural glass for shower screens."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "HASHED"
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    glass = nt.nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["Roughness"].default_value = roughness
    glass.inputs["IOR"].default_value = ior
    glass.inputs["Color"].default_value = (0.88, 0.94, 0.95, 1.0)
    nt.links.new(glass.outputs[0], out.inputs["Surface"])
    return mat


def mirror_material(name, roughness=0.02):
    """Front-silvered cabinet / medicine-cabinet mirror."""
    return solid_material(name, (0.90, 0.91, 0.93), roughness=roughness, metallic=1.0)


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
