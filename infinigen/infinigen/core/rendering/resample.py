# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import logging
import math

import bpy
import gin

from infinigen.assets.lighting import sky_lighting
from infinigen.assets.objects import rocks, trees
from infinigen.assets.paint_patch_plane import refresh_paint_patch_materials
from infinigen.assets.paint_run_plane import refresh_paint_run_materials
from infinigen.assets.wall_bubble_plane import (
    refresh_ceiling_bubble_materials,
    refresh_wall_bubble_materials,
)
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_utils import resample_node_group
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.rendering.surface_detail import (
    add_edge_bevels,
    apply_surface_imperfections,
    disable_crack_plane_lighting,
)
from infinigen.core.util.logging import Timer
from infinigen.core.util.math import FixedSeed, int_hash

logger = logging.getLogger(__name__)


def resample_all(factory_class):
    if "placeholders" not in bpy.data.collections:
        return
    for placeholder_col in bpy.data.collections["placeholders"].children:
        if "(" not in placeholder_col.name:
            continue
        classname, _ = placeholder_col.name.split("(", 1)
        if classname != factory_class.__name__:
            continue

        placeholders = [o for o in placeholder_col.objects if o.parent is None]
        for pholder in placeholders:
            factory_class.quickly_resample(pholder)


def resample_room_surfaces(scene_seed):
    """Re-apply wall, floor, and ceiling materials using a new seed.

    This lets the render step produce different material appearances for
    the same apartment geometry.  The collections written by split_rooms
    during coarse generation are used to locate the meshes.
    """
    from infinigen.core.constraints.example_solver.room import decorate as room_dec
    from infinigen.core.constraints.constraint_language.constants import RoomConstants

    constants = RoomConstants()

    wall_col = bpy.data.collections.get("unique_assets:room_wall")
    floor_col = bpy.data.collections.get("unique_assets:room_floor")
    ceiling_col = bpy.data.collections.get("unique_assets:room_ceiling")

    if wall_col and wall_col.objects:
        with Timer("Resample wall materials"):
            room_dec.room_walls(
                list(wall_col.objects),
                constants,
                material_seed=int_hash((scene_seed, "render_walls")),
            )
        with Timer("Refresh wall-bubble materials"):
            refresh_wall_bubble_materials(list(wall_col.objects))
        with Timer("Refresh paint-run materials"):
            refresh_paint_run_materials(list(wall_col.objects))
        with Timer("Refresh paint-patch materials"):
            refresh_paint_patch_materials(list(wall_col.objects))
    if floor_col and floor_col.objects:
        with Timer("Resample floor materials"):
            room_dec.room_floors(
                list(floor_col.objects),
                material_seed=int_hash((scene_seed, "render_floors")),
            )
    if ceiling_col and ceiling_col.objects:
        with Timer("Resample ceiling materials"):
            room_dec.room_ceilings(
                list(ceiling_col.objects),
                material_seed=int_hash((scene_seed, "render_ceilings")),
            )
        with Timer("Refresh ceiling-bubble materials"):
            refresh_ceiling_bubble_materials(list(ceiling_col.objects))


def resample_scene(scene_seed):
    with FixedSeed(scene_seed), Timer("Resample noise nodes in materials"):
        for material in bpy.data.materials:
            nw = NodeWrangler(material.node_tree)
            resample_node_group(nw, scene_seed)

    with FixedSeed(scene_seed), Timer("Resample noise nodes in scatters"):
        for obj in bpy.data.objects:
            for modifier in obj.modifiers:
                if not any(
                    obj.name.startswith(s)
                    for s in ["BlenderRockFactory", "CloudFactory"]
                ):
                    if modifier.type == "NODES":
                        nw = NodeWrangler(modifier.node_group)
                        resample_node_group(nw, scene_seed)

    with (
        FixedSeed(scene_seed),
        Timer("Resample all placeholders"),
    ):  # CloudFactory too expensive
        resample_all(rocks.GlowingRocksFactory)
        resample_all(trees.TreeFactory)
        resample_all(trees.BushFactory)
        # resample_all(CreatureFactory)
    with FixedSeed(scene_seed):
        sky_lighting.add_lighting()

    resample_room_surfaces(scene_seed)


_WET_ROOM_TOKENS = ("bathroom", "restroom", "kitchen")


def _is_wet_room_object(obj) -> bool:
    name = (obj.name or "").lower()
    return any(tok in name for tok in _WET_ROOM_TOKENS)


@gin.configurable
def desaturate_surface_materials(
    saturation_scale=1.0,
    collections=("unique_assets:room_wall", "unique_assets:room_ceiling"),
    skip_wet_rooms=True,
):
    """Mix dry-room wall/ceiling colors toward gray. Leaves kitchen/bathroom tiles alone."""
    if saturation_scale is None or abs(float(saturation_scale) - 1.0) < 1e-6:
        return

    fac = max(0.0, min(1.0, 1.0 - float(saturation_scale)))
    seen = set()
    for col_name in collections:
        col = bpy.data.collections.get(col_name)
        if col is None:
            continue
        for obj in col.objects:
            if obj.type != "MESH" or not obj.data.materials:
                continue
            if skip_wet_rooms and _is_wet_room_object(obj):
                continue
            for mat in obj.data.materials:
                if mat is None or mat.name in seen or mat.node_tree is None:
                    continue
                seen.add(mat.name)
                _mix_principled_base_toward_gray(mat, fac)


def _mix_principled_base_toward_gray(material, fac: float):
    nw = NodeWrangler(material.node_tree)
    for node in list(nw.nodes):
        if node.bl_idname != Nodes.PrincipledBSDF:
            continue
        base = node.inputs.get("Base Color")
        if base is None:
            continue
        gray = (0.62, 0.60, 0.56, 1.0)
        if base.links:
            src = base.links[0].from_socket
            mix = nw.new_node(
                Nodes.MixRGB,
                input_kwargs={"Fac": fac, "Color1": src, "Color2": gray},
            )
            material.node_tree.links.new(mix.outputs[0], base)
        else:
            r, g, b, a = base.default_value
            base.default_value = (
                r * (1 - fac) + gray[0] * fac,
                g * (1 - fac) + gray[1] * fac,
                b * (1 - fac) + gray[2] * fac,
                a,
            )


def _clamp_principled_socket(node, name, minimum=None, maximum=None):
    sock = node.inputs.get(name)
    if sock is None:
        return
    if sock.links:
        return
    val = float(sock.default_value)
    if minimum is not None:
        val = max(val, minimum)
    if maximum is not None:
        val = min(val, maximum)
    sock.default_value = val


def _raise_linked_roughness(material, node, minimum):
    sock = node.inputs.get("Roughness")
    if sock is None or not sock.links:
        _clamp_principled_socket(node, "Roughness", minimum=minimum)
        return
    src = sock.links[0].from_socket
    # Avoid stacking MAXIMUM nodes if realism already ran in this session.
    if src.node.bl_idname == Nodes.Math and src.node.operation == "MAXIMUM":
        return
    nw = NodeWrangler(material.node_tree)
    raised = nw.new_node(
        Nodes.Math,
        input_kwargs={0: src, 1: float(minimum)},
        attrs={"operation": "MAXIMUM"},
    )
    material.node_tree.links.new(raised.outputs[0], sock)


@gin.configurable
def dull_interior_floors(
    enabled=False,
    min_roughness=0.42,
    max_coat=0.12,
    max_specular=0.34,
):
    """Kill wet-plastic hardwood on existing blends (works without resample)."""
    if not enabled:
        return
    col = bpy.data.collections.get("unique_assets:room_floor")
    if col is None:
        return
    seen = set()
    for obj in col.objects:
        if obj.type != "MESH" or not obj.data.materials:
            continue
        for mat in obj.data.materials:
            if mat is None or mat.name in seen or mat.node_tree is None:
                continue
            seen.add(mat.name)
            for node in mat.node_tree.nodes:
                if node.bl_idname != Nodes.PrincipledBSDF:
                    continue
                _raise_linked_roughness(mat, node, min_roughness)
                _clamp_principled_socket(node, "Coat Weight", maximum=max_coat)
                _clamp_principled_socket(
                    node, "Specular IOR Level", maximum=max_specular
                )


@gin.configurable
def soften_existing_lights(
    enabled=False,
    sun_angle_deg=5.0,
    sun_energy=0.42,
    point_soft_size=0.16,
    point_energy_scale=1.2,
):
    """Soften hard sun patches and give ceiling lamps a larger, warmer pool."""
    if not enabled:
        return
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        data = obj.data
        if data.type == "SUN":
            data.angle = math.radians(float(sun_angle_deg))
            data.energy = float(sun_energy)
        elif data.type == "POINT":
            data.shadow_soft_size = max(data.shadow_soft_size, float(point_soft_size))
            data.energy *= float(point_energy_scale)


@gin.configurable
def configure_photo_cycles(
    enabled=False,
    use_ao=True,
    ao_factor=0.42,
    ao_distance=1.35,
    diffuse_bounces=8,
    glossy_bounces=5,
    transmission_bounces=8,
    max_bounces=12,
    view_transform="AgX",
    look="None",
):
    """Phone-like Cycles: bounce fill, corner AO, AgX/Filmic tone map."""
    if not enabled:
        return
    cycles = bpy.context.scene.cycles
    cycles.max_bounces = int(max_bounces)
    cycles.diffuse_bounces = int(diffuse_bounces)
    cycles.glossy_bounces = int(glossy_bounces)
    cycles.transmission_bounces = int(transmission_bounces)

    world = bpy.context.scene.world
    if world is not None and use_ao:
        # Cycles 4.x dropped world-level ambient occlusion. Fast GI is not a
        # substitute - it adds ambient light and washes the room out. Corner
        # darkening already comes from the AO node inside shader_plaster, so on
        # 4.x there is simply nothing to set here.
        light_settings = world.light_settings
        if hasattr(light_settings, "use_ambient_occlusion"):
            light_settings.use_ambient_occlusion = True
            light_settings.ao_factor = float(ao_factor)
            light_settings.distance = float(ao_distance)
        else:
            logger.debug("world AO unavailable on this Blender; relying on shader AO")

    # Assign and catch, rather than checking membership first. Both of these
    # enums are populated from the OCIO config at runtime, so
    # `bl_rna.properties[...].enum_items` reports only ('NONE',) - the old
    # membership test never matched, and the tone map was silently left at
    # whatever the scene default happened to be while the Look was never
    # applied at all. The view transform has to be set before the look,
    # because which looks exist depends on it.
    view = bpy.context.scene.view_settings
    for candidate in (view_transform, "AgX", "Filmic"):
        if not candidate:
            continue
        try:
            view.view_transform = candidate
            break
        except TypeError:
            logger.debug("view transform %r unavailable", candidate)
    if look:
        try:
            view.look = look
        except TypeError:
            logger.warning(
                "look %r unavailable for view transform %r; left at %r "
                "(4.x names them 'AgX - Base Contrast', not 'Base Contrast')",
                look, view.view_transform, view.look,
            )


def _looks_camera_invisible(mat) -> bool:
    if mat is None or mat.node_tree is None:
        return False
    types = {n.bl_idname for n in mat.node_tree.nodes}
    return Nodes.LightPath in types and Nodes.TransparentBSDF in types


@gin.configurable
def restore_ceiling_fixture_visibility(enabled=False):
    """Existing blends hide fixture meshes from camera; put a satin disc back."""
    if not enabled:
        return
    mat = bpy.data.materials.get("SynDefectCeilingFixture")
    if mat is None:
        mat = bpy.data.materials.new("SynDefectCeilingFixture")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0.86, 0.85, 0.82, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.55
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.22
        nt.links.new(bsdf.outputs[0], out.inputs[0])

    for obj in bpy.data.objects:
        if obj.type != "MESH" or "CeilingLight" not in obj.name:
            continue
        obj.visible_camera = True
        slots = obj.data.materials
        if not slots:
            slots.append(mat)
            continue
        current = slots[0]
        if current is None or _looks_camera_invisible(current):
            slots[0] = mat


@gin.configurable
def hide_cable_trunks(enabled=True):
    """Hide PVC trunking on re-render of older blends."""
    if not enabled:
        return
    tokens = (
        "WallCableTrunk",
        "CeilingCableTrunk",
        "TrunkJunction",
        "WallCableRiser",
        "CableTrunk",
    )
    n = 0
    col = bpy.data.collections.get("unique_assets:cable_trunks")
    if col is not None:
        col.hide_render = True
        col.hide_viewport = True
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not any(tok in obj.name for tok in tokens):
            continue
        obj.hide_render = True
        obj.hide_viewport = True
        n += 1
    if n:
        logger.info("Hid %s cable-trunk objects", n)


_FIXTURE_NAME_TOKENS = (
    "Bathtub",
    "BathroomSink",
    "StandingSink",
    "ToiletFactory",
)


@gin.configurable
def flatten_bathroom_fixture_materials(enabled=True):
    """Drop heavy ceramic displacement on tubs/sinks/toilets (helps existing blends)."""
    if not enabled:
        return
    mats = set()
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not any(tok in obj.name for tok in _FIXTURE_NAME_TOKENS):
            continue
        for slot in obj.material_slots:
            if slot.material is not None:
                mats.add(slot.material)
        if obj.active_material is not None:
            mats.add(obj.active_material)
    n = 0
    for mat in mats:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        out = next(
            (nd for nd in mat.node_tree.nodes if nd.type == "OUTPUT_MATERIAL"),
            None,
        )
        if out is None or "Displacement" not in out.inputs:
            continue
        links = list(out.inputs["Displacement"].links)
        if not links:
            continue
        for link in links:
            mat.node_tree.links.remove(link)
        if hasattr(mat, "displacement_method"):
            mat.displacement_method = "BUMP"
        n += 1
    if n:
        logger.info("Flattened displacement on %s bathroom fixture materials", n)


@gin.configurable
def apply_realism_adjustments(
    refresh_room_surfaces=False,
    room_surface_seed=17,
    scene_seed=None,
    resample_idx=None,
):
    """Optional render-time realism hooks. All default to no-op unless gin-enabled."""
    already_resampled = resample_idx is not None and resample_idx != 0
    if refresh_room_surfaces and not already_resampled:
        seed = scene_seed if scene_seed is not None else room_surface_seed
        resample_room_surfaces(int_hash((int(seed), "realism_surfaces")))
        sky_lighting.add_lighting()
    desaturate_surface_materials()
    dull_interior_floors()
    flatten_bathroom_fixture_materials()
    hide_cable_trunks()
    soften_existing_lights()
    add_edge_bevels()
    apply_surface_imperfections()
    disable_crack_plane_lighting()
    configure_photo_cycles()
    restore_ceiling_fixture_visibility()
    sky_lighting.add_camera_based_lighting()
