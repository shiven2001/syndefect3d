# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
# Authors: David Yan


import argparse
import logging
import math
import shutil
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bpy
import gin
import numpy as np
import trimesh

from infinigen.core.util import blender as butil

try:
    import coacd
except ImportError:
    coacd = None
    warnings.warn("coacd could not be imported. Some features may be unavailable.")

FORMAT_CHOICES = ["fbx", "obj", "usdc", "usda", "stl", "ply"]
BAKE_TYPES = {
    "DIFFUSE": "Base Color",
    "ROUGHNESS": "Roughness",
    "NORMAL": "Normal",
}  # 'EMIT':'Emission Color' #  "GLOSSY": 'Specular IOR Level', 'TRANSMISSION':'Transmission Weight' don't export
SPECIAL_BAKE = {"METAL": "Metallic", "TRANSMISSION": "Transmission Weight"}
ALPHA_BAKE = {"ALPHA": "Alpha"}  # Alpha uses EMIT baking like SPECIAL_BAKE
ALL_BAKE = BAKE_TYPES | SPECIAL_BAKE | ALPHA_BAKE


def apply_all_modifiers(obj):
    for mod in obj.modifiers:
        if mod is None:
            continue
        try:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=mod.name)
            logging.info(f"Applied modifier {mod} on {obj}")
            obj.select_set(False)
        except RuntimeError:
            logging.info(f"Can't apply {mod} on {obj}")
            obj.select_set(False)
            return


def realizeInstances(obj):
    for mod in obj.modifiers:
        if mod is None or mod.type != "NODES":
            continue
        geo_group = mod.node_group
        outputNode = geo_group.nodes["Group Output"]

        logging.info(f"Realizing instances on {mod}")
        link = outputNode.inputs[0].links[0]
        from_socket = link.from_socket
        geo_group.links.remove(link)
        realizeNode = geo_group.nodes.new(type="GeometryNodeRealizeInstances")
        geo_group.links.new(realizeNode.inputs[0], from_socket)
        geo_group.links.new(outputNode.inputs[0], realizeNode.outputs[0])


def remove_shade_smooth(obj):
    for mod in obj.modifiers:
        if mod is None or mod.type != "NODES":
            continue
        geo_group = mod.node_group
        outputNode = geo_group.nodes["Group Output"]
        if geo_group.nodes.get("Set Shade Smooth"):
            logging.info("Removing shade smooth on " + obj.name)
            smooth_node = geo_group.nodes["Set Shade Smooth"]
        else:
            continue

        link = smooth_node.inputs[0].links[0]
        from_socket = link.from_socket
        geo_group.links.remove(link)
        geo_group.links.new(outputNode.inputs[0], from_socket)


def check_material_geonode(node_tree):
    if node_tree.nodes.get("Set Material"):
        logging.info("Found set material!")
        return True

    for node in node_tree.nodes:
        if node.type == "GROUP" and check_material_geonode(node.node_tree):
            return True

    return False


def handle_geo_modifiers(obj, export_usd):
    has_geo_nodes = False
    for mod in obj.modifiers:
        if mod is None or mod.type != "NODES":
            continue
        has_geo_nodes = True

    if has_geo_nodes and not obj.data.materials:
        mat = bpy.data.materials.new(name=f"{mod.name} shader")
        obj.data.materials.append(mat)
        mat.use_nodes = True
        mat.node_tree.nodes.remove(mat.node_tree.nodes["Principled BSDF"])

    if not export_usd:
        realizeInstances(obj)


def split_glass_mats():
    split_objs = []
    for obj in bpy.data.objects:
        if obj.hide_render or obj.hide_viewport:
            continue
        if any(
            exclude in obj.name
            for exclude in ["BowlFactory", "CupFactory", "OvenFactory", "BottleFactory"]
        ):
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            if ("shader_glass" in mat.name or "shader_lamp_bulb" in mat.name) and len(
                obj.material_slots
            ) >= 2:
                logging.info(f"Splitting {obj}")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.separate(type="MATERIAL")
                bpy.ops.object.mode_set(mode="OBJECT")
                obj.select_set(False)
                split_objs.append(obj.name)
                break

    matches = [
        obj
        for split_obj in split_objs
        for obj in bpy.data.objects
        if split_obj in obj.name
    ]
    for match in matches:
        if len(match.material_slots) == 0 or match.material_slots[0] is None:
            continue
        mat = match.material_slots[0].material
        if mat is None:
            continue
        if "shader_glass" in mat.name or "shader_lamp_bulb" in mat.name:
            match.name = f"{match.name}_SPLIT_GLASS"


def clean_names(obj=None):
    if obj is not None:
        obj.name = (obj.name).replace(" ", "_")
        obj.name = (obj.name).replace(".", "_")

        if obj.type == "MESH":
            for uv_map in obj.data.uv_layers:
                uv_map.name = uv_map.name.replace(".", "_")

        for mat in bpy.data.materials:
            if mat is None:
                continue
            mat.name = (mat.name).replace(" ", "_")
            mat.name = (mat.name).replace(".", "_")

        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            mat.name = (mat.name).replace(" ", "_")
            mat.name = (mat.name).replace(".", "_")
        return

    for obj in bpy.data.objects:
        obj.name = (obj.name).replace(" ", "_")
        obj.name = (obj.name).replace(".", "_")

        if obj.type == "MESH":
            for uv_map in obj.data.uv_layers:
                uv_map.name = uv_map.name.replace(
                    ".", "_"
                )  # if uv has '.' in name the node will export wrong in USD

    for mat in bpy.data.materials:
        if mat is None:
            continue
        mat.name = (mat.name).replace(" ", "_")
        mat.name = (mat.name).replace(".", "_")


def remove_obj_parents(obj=None):
    if obj is not None:
        old_location = obj.matrix_world.to_translation()
        obj.parent = None
        obj.matrix_world.translation = old_location
        return

    for obj in bpy.data.objects:
        old_location = obj.matrix_world.to_translation()
        obj.parent = None
        obj.matrix_world.translation = old_location


def delete_objects():
    logging.info("Deleting placeholders collection")
    collection_name = "placeholders"
    collection = bpy.data.collections.get(collection_name)

    if collection:
        for scene in bpy.data.scenes:
            if collection.name in scene.collection.children:
                scene.collection.children.unlink(collection)

        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        def delete_child_collections(parent_collection):
            for child_collection in parent_collection.children:
                delete_child_collections(child_collection)
                bpy.data.collections.remove(child_collection)

        delete_child_collections(collection)
        bpy.data.collections.remove(collection)

    if bpy.data.objects.get("Grid"):
        bpy.data.objects.remove(bpy.data.objects["Grid"], do_unlink=True)

    if bpy.data.objects.get("atmosphere"):
        bpy.data.objects.remove(bpy.data.objects["atmosphere"], do_unlink=True)

    if bpy.data.objects.get("KoleClouds"):
        bpy.data.objects.remove(bpy.data.objects["KoleClouds"], do_unlink=True)


def rename_all_meshes(obj=None):
    if obj is not None:
        if obj.data and obj.data.users == 1:
            obj.data.name = obj.name
        return

    for obj in bpy.data.objects:
        if obj.data and obj.data.users == 1:
            obj.data.name = obj.name


def update_visibility():
    outliner_area = next(a for a in bpy.context.screen.areas if a.type == "OUTLINER")
    space = outliner_area.spaces[0]
    space.show_restrict_column_viewport = True  # Global visibility (Monitor icon)
    collection_view = {}
    obj_view = {}
    for collection in bpy.data.collections:
        collection_view[collection] = collection.hide_render
        collection.hide_viewport = False  # reenables viewports for all
        collection.hide_render = False  # enables renders for all collections

    # disables viewports and renders for all objs
    for obj in bpy.data.objects:
        obj_view[obj] = obj.hide_render
        obj.hide_viewport = True
        obj.hide_render = True
        obj.hide_set(0)

    return collection_view, obj_view


def uv_unwrap(obj):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    obj.data.uv_layers.new(name="ExportUV")
    bpy.context.object.data.uv_layers["ExportUV"].active = True

    logging.info("UV Unwrapping")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.uv.smart_project(angle_limit=0.7)
    except RuntimeError:
        logging.info("UV Unwrap failed, skipping mesh")
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)
        return False
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return True


def bakeVertexColors(obj):
    logging.info(f"Baking vertex color on {obj}")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    vertColor = bpy.context.object.data.color_attributes.new(
        name="VertColor", domain="CORNER", type="BYTE_COLOR"
    )
    bpy.context.object.data.attributes.active_color = vertColor
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, target="VERTEX_COLORS")
    obj.select_set(False)


def apply_baked_tex(
    obj, paramDict={}, texture_dir=None, img_size=None, export_name=None
):
    bpy.context.view_layer.objects.active = obj
    bpy.context.object.data.uv_layers["ExportUV"].active_render = True
    for uv_layer in reversed(obj.data.uv_layers):
        if "ExportUV" not in uv_layer.name:
            logging.info(f"Removed extraneous UV Layer {uv_layer}")
            obj.data.uv_layers.remove(uv_layer)

    # Get image size from existing baked textures to match resolution
    if img_size is None:
        img_size = 1024  # default
        if texture_dir:
            texture_dir_path = Path(texture_dir)
            # Try to find an existing baked texture to get the image size
            if texture_dir_path.exists():
                for tex_file in texture_dir_path.glob("*_DIFFUSE.png"):
                    try:
                        existing_img = bpy.data.images.load(str(tex_file))
                        img_size = existing_img.size[0]
                        bpy.data.images.remove(existing_img)
                        break
                    except:
                        pass

    for slot in obj.material_slots:
        mat = slot.material
        if mat is None:
            continue
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        logging.info("Reapplying baked texs on " + mat.name)

        # Before deleting nodes, preserve alpha connection information and material settings
        # Look for mask texture or any texture connected to Alpha input
        alpha_connection_info = None
        baked_alpha_img = None
        material_alpha_settings = {}

        # Preserve material blend mode and alpha settings
        if mat:
            material_alpha_settings = {
                "blend_method": mat.blend_method,
                "alpha_threshold": mat.alpha_threshold,
                "shadow_method": mat.shadow_method,
                "show_transparent_back": mat.show_transparent_back,
            }

        if nodes.get("Principled BSDF"):
            bsdf_before = nodes["Principled BSDF"]
            alpha_input = bsdf_before.inputs.get("Alpha")
            if alpha_input and alpha_input.is_linked:
                # Find what's connected to alpha
                alpha_link = alpha_input.links[0]
                from_node = alpha_link.from_node
                from_socket = alpha_link.from_socket

                # Check if it's a mask texture (label contains "mask" or "Mask")
                # or if it's connected through an RGBToBW node (common for masks)
                # or if it's connected through a Separate Color node (common for texture_ prefixed meshes)
                if from_node.bl_idname == "ShaderNodeSeparateColor":
                    # Check if Separate Color is connected to a texture (likely a mask)
                    # For texture_ prefixed meshes: mask_img.Color -> Separate Color (RGB) -> Separate Color (Red) -> Alpha
                    separate_color_node = from_node
                    output_socket_name = from_socket.name

                    # Check if it's outputting Red channel (or R)
                    if output_socket_name in ["Red", "R"]:
                        # Trace back to find the mask image texture
                        if (
                            separate_color_node.inputs.get("Color")
                            and separate_color_node.inputs["Color"].is_linked
                        ):
                            color_input_link = separate_color_node.inputs[
                                "Color"
                            ].links[0]
                            mask_tex = color_input_link.from_node

                            if mask_tex.type == "TEX_IMAGE" and mask_tex.image:
                                # Found mask texture connected via Separate Color (Red channel)
                                alpha_connection_info = {
                                    "type": "mask_texture_separate_color_red",
                                    "image": mask_tex.image,
                                    "node_name": mask_tex.name,
                                    "separate_color_mode": (
                                        separate_color_node.mode
                                        if hasattr(separate_color_node, "mode")
                                        else None
                                    ),
                                }
                                logging.info(
                                    f"Found mask texture via Separate Color (Red): {mask_tex.name}"
                                )
                elif from_node.bl_idname == "ShaderNodeRGBToBW":
                    # Check if RGBToBW is connected to a texture (likely a mask)
                    if from_node.inputs["Color"].is_linked:
                        mask_tex = from_node.inputs["Color"].links[0].from_node
                        if mask_tex.type == "TEX_IMAGE" and mask_tex.image:
                            # RGBToBW typically used for mask textures
                            alpha_connection_info = {
                                "type": "mask_texture_rgb2bw",
                                "image": mask_tex.image,
                                "node_name": mask_tex.name,
                            }
                            logging.info(
                                f"Found mask texture via RGBToBW: {mask_tex.name}"
                            )
                elif from_node.type == "TEX_IMAGE" and from_node.image:
                    node_label = from_node.label.lower() if from_node.label else ""
                    node_name = from_node.name.lower()
                    if "mask" in node_label or "mask" in node_name:
                        # This is a mask texture - preserve it
                        alpha_connection_info = {
                            "type": "mask_texture",
                            "image": from_node.image,
                            "node_name": from_node.name,
                        }
                        logging.info(f"Found mask texture for alpha: {from_node.name}")
                    elif from_socket.name == "Alpha":
                        # Opacity texture connected via Alpha output
                        if "opacity" in node_label or "alpha" in node_label:
                            alpha_connection_info = {
                                "type": "opacity_texture",
                                "image": from_node.image,
                                "node_name": from_node.name,
                                "use_alpha_channel": True,
                            }
                            logging.info(
                                f"Found opacity texture for alpha: {from_node.name}"
                            )
                    else:
                        # Check if this texture is connected to Alpha but not through RGBToBW
                        # This might be a direct mask connection
                        # Look at the image filename to see if it's likely a mask
                        img_name = (
                            from_node.image.name.lower() if from_node.image.name else ""
                        )
                        if "mask" in img_name:
                            alpha_connection_info = {
                                "type": "mask_texture",
                                "image": from_node.image,
                                "node_name": from_node.name,
                            }
                            logging.info(
                                f"Found mask texture (by image name) for alpha: {from_node.name}"
                            )

        # We'll use the DIFFUSE baked texture's Red channel for alpha instead of baking separately

        # delete all nodes except baked nodes and bsdf
        excludedNodes = [type + "_node" for type in ALL_BAKE]
        excludedNodes.extend(
            ["ALPHA_node", "Material Output", "Principled BSDF"]
        )  # Include ALPHA_node
        for n in nodes:
            if n.name not in excludedNodes:
                nodes.remove(
                    n
                )  # deletes an arbitrary principled BSDF in the case of a mix, which is handled below

        output = nodes["Material Output"]

        # stick baked texture in material
        if nodes.get("Principled BSDF") is None:  # no bsdf
            logging.info("No BSDF, creating new one")
            principled_bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
        elif (
            len(output.inputs[0].links) != 0
            and output.inputs[0].links[0].from_node.bl_idname
            == "ShaderNodeBsdfPrincipled"
        ):  # trivial bsdf graph
            logging.info("Trivial shader graph, using old BSDF")
            principled_bsdf_node = nodes["Principled BSDF"]
        else:
            logging.info("Non-trivial shader graph, creating new BSDF")
            nodes.remove(nodes["Principled BSDF"])  # shader graph was a mix of bsdfs
            principled_bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")

        links = mat.node_tree.links

        # create the new shader node links
        links.new(output.inputs[0], principled_bsdf_node.outputs[0])
        for type in ALL_BAKE:
            if not nodes.get(type + "_node"):
                continue
            tex_node = nodes[type + "_node"]
            if type == "NORMAL":
                normal_node = nodes.new("ShaderNodeNormalMap")
                links.new(normal_node.inputs["Color"], tex_node.outputs[0])
                links.new(
                    principled_bsdf_node.inputs[ALL_BAKE[type]], normal_node.outputs[0]
                )
                continue
            # For Base Color, always connect Color output explicitly to avoid alpha connection
            if type == "DIFFUSE":
                # Explicitly use Color output to prevent alpha from being connected
                links.new(
                    principled_bsdf_node.inputs[ALL_BAKE[type]],
                    tex_node.outputs["Color"],
                )
            else:
                links.new(
                    principled_bsdf_node.inputs[ALL_BAKE[type]], tex_node.outputs[0]
                )

        # Restore alpha connection using baked ALPHA texture (baked the same way as other textures)
        if alpha_connection_info and nodes.get("ALPHA_node"):
            alpha_node = nodes["ALPHA_node"]

            # The ALPHA texture was already baked, so we can use it directly
            # Extract Red channel for alpha (matching the original mask texture pattern)
            separate_color_node = nodes.new("ShaderNodeSeparateColor")
            separate_color_node.mode = "RGB"

            # Connect ALPHA texture Color output to Separate Color input
            links.new(alpha_node.outputs["Color"], separate_color_node.inputs["Color"])
            # Connect Separate Color Red output to Principled BSDF Alpha input
            links.new(
                separate_color_node.outputs["Red"], principled_bsdf_node.inputs["Alpha"]
            )
            logging.info(
                f"Connected baked ALPHA texture Red channel to Alpha for {mat.name}"
            )

        # bring back cleared param values
        if mat.name in paramDict:
            principled_bsdf_node.inputs["Metallic"].default_value = paramDict[mat.name][
                "Metallic"
            ]
            principled_bsdf_node.inputs["Sheen Weight"].default_value = paramDict[
                mat.name
            ]["Sheen Weight"]
            principled_bsdf_node.inputs["Coat Weight"].default_value = paramDict[
                mat.name
            ]["Coat Weight"]

        # Restore material alpha/transparency settings if they were preserved
        # Set opacity threshold to 0.5 for all materials with alpha (like roughness threshold)
        if alpha_connection_info:
            # Set blend method and alpha threshold for proper transparency in USD/Isaac Sim
            mat.blend_method = "CLIP"
            mat.alpha_threshold = 0.5  # Set to 0.5 for all
            mat.shadow_method = "CLIP"
            mat.show_transparent_back = (
                material_alpha_settings.get("show_transparent_back", False)
                if material_alpha_settings
                else False
            )

            # Set custom properties that might be exported to USD (Isaac Sim might read these)
            mat["opacityThreshold"] = 0.5
            mat["opacity_threshold"] = 0.5
            mat["usd_opacity_threshold"] = 0.5

            # Also set on Principled BSDF node if possible (some USD exporters read node properties)
            if nodes.get("Principled BSDF"):
                bsdf_node = nodes["Principled BSDF"]
                # Try to set any opacity-related properties on the node
                if hasattr(bsdf_node, "inputs") and "Alpha" in bsdf_node.inputs:
                    # The Alpha input itself is already connected, but we can ensure threshold is set
                    pass  # Alpha connection is already handled above

            logging.info(
                f"Set opacity threshold to 0.5 for {obj.name}: blend_method={mat.blend_method}, alpha_threshold={mat.alpha_threshold}"
            )


def create_glass_shader(node_tree, export_usd):
    nodes = node_tree.nodes
    if nodes.get("Glass BSDF"):
        color = nodes["Glass BSDF"].inputs[0].default_value
        roughness = nodes["Glass BSDF"].inputs[1].default_value
        ior = nodes["Glass BSDF"].inputs[2].default_value

    if nodes.get("Principled BSDF"):
        nodes.remove(nodes["Principled BSDF"])

    principled_bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")

    if nodes.get("Glass BSDF"):
        principled_bsdf_node.inputs["Base Color"].default_value = color
        principled_bsdf_node.inputs["Roughness"].default_value = roughness
        principled_bsdf_node.inputs["IOR"].default_value = ior
    else:
        principled_bsdf_node.inputs["Roughness"].default_value = 0

    principled_bsdf_node.inputs["Transmission Weight"].default_value = 1
    if export_usd:
        principled_bsdf_node.inputs["Alpha"].default_value = 0
    node_tree.links.new(
        principled_bsdf_node.outputs[0], nodes["Material Output"].inputs[0]
    )


def process_glass_materials(obj, export_usd):
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        outputNode = nodes["Material Output"]
        if nodes.get("Glass BSDF"):
            if (
                outputNode.inputs[0].links[0].from_node.bl_idname
                == "ShaderNodeBsdfGlass"
            ):
                logging.info(f"Creating glass material on {obj.name}")
            else:
                logging.info(
                    f"Non-trivial glass material on {obj.name}, material export will be inaccurate"
                )
            create_glass_shader(mat.node_tree, export_usd)
        elif "glass" in mat.name or "shader_lamp_bulb" in mat.name:
            logging.info(f"Creating glass material on {obj.name}")
            create_glass_shader(mat.node_tree, export_usd)


def bake_pass(obj, dest: Path, img_size, bake_type, export_usd, export_name=None):
    if export_name is None:
        img = bpy.data.images.new(f"{obj.name}_{bake_type}", img_size, img_size)
        clean_name = (obj.name).replace(" ", "_").replace(".", "_").replace("/", "_")
        file_path = dest / f"{clean_name}_{bake_type}.png"
    else:
        img = bpy.data.images.new(f"{export_name}_{bake_type}", img_size, img_size)
        file_path = dest / f"{export_name}_{bake_type}.png"
    dest = dest / "textures"

    bake_obj = False
    bake_exclude_mats = {}

    # materials are stored as stack so when removing traverse the reversed list
    for index, slot in reversed(list(enumerate(obj.material_slots))):
        mat = slot.material
        if mat is None:
            bpy.context.object.active_material_index = index
            bpy.ops.object.material_slot_remove()
            continue

        logging.info(mat.name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes

        output = nodes["Material Output"]

        img_node = nodes.new("ShaderNodeTexImage")
        img_node.name = f"{bake_type}_node"
        img_node.image = img
        img_node.select = True
        nodes.active = img_node
        img_node.select = True

        if len(output.inputs["Displacement"].links) != 0:
            bake_obj = True

        if len(output.inputs[0].links) == 0:
            logging.info(f"{mat.name} has no surface output, not using baked textures")
            bake_exclude_mats[mat] = img_node
            continue

        # surface_node = output.inputs[0].links[0].from_node
        # if (
        #     bake_type in ALL_BAKE
        #     and surface_node.bl_idname == "ShaderNodeBsdfPrincipled"
        #     and len(surface_node.inputs[ALL_BAKE[bake_type]].links) == 0
        # ):  # trivial bsdf graph
        #     logging.info(
        #         f"{mat.name} has no procedural input for {bake_type}, not using baked textures"
        #     )
        #     bake_exclude_mats[mat] = img_node
        #     continue

        bake_obj = True

    if bake_type in SPECIAL_BAKE:
        internal_bake_type = "EMIT"
    else:
        internal_bake_type = bake_type

    if bake_obj:
        logging.info(f"Baking {bake_type} pass")
        bpy.ops.object.bake(
            type=internal_bake_type, pass_filter={"COLOR"}, save_mode="EXTERNAL"
        )
        img.filepath_raw = str(file_path)
        img.save()
        logging.info(f"Saving to {file_path}")
    else:
        logging.info(f"No necessary materials to bake on {obj.name}, skipping bake")

    for mat, img_node in bake_exclude_mats.items():
        mat.node_tree.nodes.remove(img_node)


def bake_alpha_pass(obj, dest, img_size, export_usd, export_name=None):
    """
    Bake the Alpha input from Principled BSDF to an image, using the same approach as other textures.
    This follows the pattern of bake_pass but uses EMIT baking like SPECIAL_BAKE.
    """
    # Check if any material has an alpha connection
    has_alpha = False
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        if nodes.get("Principled BSDF"):
            bsdf = nodes["Principled BSDF"]
            alpha_input = bsdf.inputs.get("Alpha")
            if alpha_input and alpha_input.is_linked:
                has_alpha = True
                break

    if not has_alpha:
        logging.info(f"No alpha connection found on {obj.name}, skipping alpha bake")
        return

    # Create image for baked alpha (same pattern as bake_pass)
    if export_name is None:
        img = bpy.data.images.new(f"{obj.name}_ALPHA", img_size, img_size)
        clean_name = (obj.name).replace(" ", "_").replace(".", "_").replace("/", "_")
        file_path = dest / "textures" / f"{clean_name}_ALPHA.png"
    else:
        img = bpy.data.images.new(f"{export_name}_ALPHA", img_size, img_size)
        file_path = dest / "textures" / f"{export_name}_ALPHA.png"

    # Store original links to restore later
    links_to_restore = []

    # For each material, temporarily connect Alpha to emission for baking
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue

        nodes = mat.node_tree.nodes
        output = nodes.get("Material Output")
        bsdf = nodes.get("Principled BSDF")

        if not output or not bsdf:
            continue

        alpha_input = bsdf.inputs.get("Alpha")
        if not alpha_input or not alpha_input.is_linked:
            continue

        # Create image node for this material
        img_node = nodes.new("ShaderNodeTexImage")
        img_node.name = "ALPHA_node"
        img_node.image = img
        img_node.select = True
        nodes.active = img_node

        # Get what's connected to Alpha
        links = mat.node_tree.links
        alpha_source = alpha_input.links[0].from_socket

        # Temporarily connect Alpha source to emission
        emission = nodes.new("ShaderNodeEmission")
        links.new(alpha_source, emission.inputs["Color"])

        # Temporarily connect emission to surface output
        # Store original surface connection BEFORE removing it
        original_surface_links = list(output.inputs["Surface"].links)
        original_from_socket = None
        if original_surface_links:
            original_from_socket = original_surface_links[
                0
            ].from_socket  # Store socket before removing link
            links.remove(original_surface_links[0])
        links.new(emission.outputs["Emission"], output.inputs["Surface"])

        # Store for restoration (store socket references, not link objects)
        links_to_restore.append((mat, nodes, output, original_from_socket, emission))

    if links_to_restore:
        # Ensure ExportUV is active
        if "ExportUV" in obj.data.uv_layers:
            obj.data.uv_layers["ExportUV"].active_render = True

        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Bake using EMIT (same as SPECIAL_BAKE)
        logging.info(f"Baking ALPHA pass for {obj.name}")
        bpy.ops.object.bake(type="EMIT", pass_filter={"COLOR"}, save_mode="EXTERNAL")

        # Save the baked image
        file_path.parent.mkdir(parents=True, exist_ok=True)
        img.filepath_raw = str(file_path)
        img.save()
        logging.info(f"Saved baked alpha texture to: {file_path}")

        # Restore original connections
        for (
            mat,
            nodes,
            output,
            original_from_socket,
            emission,
        ) in links_to_restore:
            links = mat.node_tree.links

            # Remove emission connection
            emission_links = [
                l for l in output.inputs["Surface"].links if l.from_node == emission
            ]
            for link in emission_links:
                links.remove(link)

            # Restore original surface connection if it existed
            if original_from_socket:
                try:
                    links.new(original_from_socket, output.inputs["Surface"])
                except Exception as e:
                    logging.warning(f"Could not restore original surface link: {e}")

            # Clean up emission node
            if emission.name in nodes:
                nodes.remove(nodes[emission.name])


def bake_special_emit(obj, dest, img_size, export_usd, bake_type, export_name=None):
    # If at least one material has both a BSDF and non-zero bake type value, then bake
    should_bake = False

    # (Root node, From Socket, To Socket)
    links_removed = []
    links_added = []

    for slot in obj.material_slots:
        mat = slot.material
        if mat is None:
            logging.warn("No material on mesh, skipping...")
            continue
        if not mat.use_nodes:
            logging.warn("Material has no nodes, skipping...")
            continue

        nodes = mat.node_tree.nodes
        principled_bsdf_node = None
        root_node = None
        logging.info(f"{mat.name} has {len(nodes)} nodes: {nodes}")
        for node in nodes:
            if node.type != "GROUP":
                continue

            for subnode in node.node_tree.nodes:
                logging.info(f" [{subnode.type}] {subnode.name} {subnode.bl_idname}")
                if subnode.type == "BSDF_PRINCIPLED":
                    logging.debug(f" BSDF_PRINCIPLED: {subnode.inputs}")
                    principled_bsdf_node = subnode
                    root_node = node

        if nodes.get("Principled BSDF"):
            principled_bsdf_node = nodes["Principled BSDF"]
            root_node = mat
        elif not principled_bsdf_node:
            logging.warn("No Principled BSDF, skipping...")
            continue
        elif ALL_BAKE[bake_type] not in principled_bsdf_node.inputs:
            logging.warn(f"No {bake_type} input, skipping...")
            continue

        # Here, we've found the proper BSDF and bake type input. Set up the scene graph
        # for baking.
        outputSoc = principled_bsdf_node.outputs[0].links[0].to_socket

        # Remove the BSDF link to Output first
        l = principled_bsdf_node.outputs[0].links[0]
        from_socket, to_socket = l.from_socket, l.to_socket
        logging.debug(f"Removing link: {from_socket.name} => {to_socket.name}")
        root_node.node_tree.links.remove(l)
        links_removed.append((root_node, from_socket, to_socket))

        # Get bake_type value
        bake_input = principled_bsdf_node.inputs[ALL_BAKE[bake_type]]
        bake_val = bake_input.default_value
        logging.info(f"{bake_type} value: {bake_val}")

        if bake_val > 0:
            should_bake = True

        # Make a color input matching the metallic value
        col = root_node.node_tree.nodes.new("ShaderNodeRGB")
        col.outputs[0].default_value = (bake_val, bake_val, bake_val, 1.0)

        # Link the color to output
        new_link = root_node.node_tree.links.new(col.outputs[0], outputSoc)
        links_added.append((root_node, col.outputs[0], outputSoc))
        logging.debug(
            f"Linking {col.outputs[0].name} to {outputSoc.name}({outputSoc.bl_idname}): {new_link}"
        )

    # After setting up all materials, bake if applicable
    if should_bake:
        bake_pass(obj, dest, img_size, bake_type, export_usd, export_name)

    # After baking, undo the temporary changes to the scene graph
    for n, from_soc, to_soc in links_added:
        logging.debug(
            f"Removing added link:\t{n.name}: {from_soc.name} => {to_soc.name}"
        )
        for l in n.node_tree.links:
            if l.from_socket == from_soc and l.to_socket == to_soc:
                n.node_tree.links.remove(l)
                logging.debug(
                    f"Removed link:\t{n.name}: {from_soc.name} => {to_soc.name}"
                )

    for n, from_soc, to_soc in links_removed:
        logging.debug(f"Adding back link:\t{n.name}: {from_soc.name} => {to_soc.name}")
        n.node_tree.links.new(from_soc, to_soc)


def remove_params(mat, node_tree):
    nodes = node_tree.nodes
    paramDict = {}
    if nodes.get("Material Output"):
        output = nodes["Material Output"]
    elif nodes.get("Group Output"):
        output = nodes["Group Output"]
    else:
        raise ValueError("Could not find material output node")

    if (
        nodes.get("Principled BSDF")
        and output.inputs[0].links[0].from_node.bl_idname == "ShaderNodeBsdfPrincipled"
    ):
        principled_bsdf_node = nodes["Principled BSDF"]
        metal = principled_bsdf_node.inputs[
            "Metallic"
        ].default_value  # store metallic value and set to 0
        sheen = principled_bsdf_node.inputs["Sheen Weight"].default_value
        clearcoat = principled_bsdf_node.inputs["Coat Weight"].default_value
        paramDict[mat.name] = {
            "Metallic": metal,
            "Sheen Weight": sheen,
            "Coat Weight": clearcoat,
        }
        principled_bsdf_node.inputs["Metallic"].default_value = 0
        principled_bsdf_node.inputs["Sheen Weight"].default_value = 0
        principled_bsdf_node.inputs["Coat Weight"].default_value = 0
        return paramDict

    for node in nodes:
        if node.type == "GROUP":
            paramDict = remove_params(mat, node.node_tree)
            if len(paramDict) != 0:
                return paramDict

    return paramDict


def process_interfering_params(obj):
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        paramDict = remove_params(mat, mat.node_tree)
    return paramDict


def skipBake(obj):
    if not obj.data.materials:
        logging.info("No material on mesh, skipping...")
        return True

    if len(obj.data.vertices) == 0:
        logging.info("Mesh has no vertices, skipping ...")
        return True

    return False


def triangulate_mesh(obj: bpy.types.Object):
    logging.debug("Triangulating Mesh")
    if obj.type == "MESH":
        view_state = obj.hide_viewport
        obj.hide_viewport = False
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        logging.debug(f"Triangulating {obj}")
        bpy.ops.mesh.quads_convert_to_tris()
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)
        obj.hide_viewport = view_state


def triangulate_meshes():
    logging.debug("Triangulating Meshes")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            view_state = obj.hide_viewport
            obj.hide_viewport = False
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            logging.debug(f"Triangulating {obj}")
            bpy.ops.mesh.quads_convert_to_tris()
            bpy.ops.object.mode_set(mode="OBJECT")
            obj.select_set(False)
            obj.hide_viewport = view_state


def adjust_wattages():
    logging.info("Adjusting light wattage")
    for obj in bpy.context.scene.objects:
        if obj.type == "LIGHT" and obj.data.type == "POINT":
            light = obj.data
            if hasattr(light, "energy") and hasattr(light, "shadow_soft_size"):
                X = light.energy
                r = light.shadow_soft_size
                # Reduced multipliers for Isaac Sim compatibility
                new_wattage = (X * 2 / (4 * math.pi)) * 100 / (4 * math.pi * r**2) * 10
                light.energy = new_wattage


def set_center_of_mass():
    logging.info("Resetting center of mass of objects")
    for obj in bpy.context.scene.objects:
        if not obj.hide_render:
            view_state = obj.hide_viewport
            obj.hide_viewport = False
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
            obj.select_set(False)
            obj.hide_viewport = view_state


def duplicate_node_groups(node_tree, group_map=None):
    if group_map is None:
        group_map = {}

    for node in node_tree.nodes:
        if node.type == "GROUP":
            group = node.node_tree
            if group not in group_map:
                group_copy = group.copy()
                group_copy.name = f"{group.name}_copy"
                group_map[group] = group_copy

                duplicate_node_groups(group_copy, group_map)
            else:
                group_copy = group_map[group]

            node.node_tree = group_copy

    return group_map


def deep_copy_material(original_material, new_name_suffix="_deepcopy"):
    new_mat = original_material.copy()
    new_mat.name = original_material.name + new_name_suffix
    if new_mat.use_nodes and new_mat.node_tree:
        duplicate_node_groups(new_mat.node_tree)
    return new_mat


def bake_object(obj, dest, img_size, export_usd, export_name=None):
    if not uv_unwrap(obj):
        return

    bpy.ops.object.select_all(action="DESELECT")

    with butil.SelectObjects(obj):
        for slot in obj.material_slots:
            mat = slot.material
            if mat is not None:
                slot.material = deep_copy_material(
                    mat
                )  # we duplicate in the case of distinct meshes sharing materials

        process_glass_materials(obj, export_usd)

        for bake_type in SPECIAL_BAKE:
            bake_special_emit(obj, dest, img_size, export_usd, bake_type, export_name)

        # bake_normals(obj, dest, img_size, export_usd)
        paramDict = process_interfering_params(obj)
        for bake_type in BAKE_TYPES:
            bake_pass(obj, dest, img_size, bake_type, export_usd, export_name)

        # Bake Alpha if there's an alpha connection (must be done before apply_baked_tex deletes nodes)
        bake_alpha_pass(obj, dest, img_size, export_usd, export_name)

        # Get textures directory path for saving mask textures
        texture_dir = (
            dest / "textures"
            if isinstance(dest, Path)
            else Path(str(dest)) / "textures"
        )
        texture_dir.mkdir(exist_ok=True)
        apply_baked_tex(
            obj,
            paramDict,
            texture_dir=texture_dir,
            img_size=img_size,
            export_name=export_name,
        )


def bake_scene(folderPath: Path, image_res, vertex_colors, export_usd):
    for obj in bpy.data.objects:
        logging.info("---------------------------")
        logging.info(obj.name)

        if obj.type != "MESH" or obj not in list(bpy.context.view_layer.objects):
            logging.info("Not mesh, skipping ...")
            continue

        if skipBake(obj):
            continue

        if format == "stl":
            continue

        obj.hide_render = False
        obj.hide_viewport = False

        if vertex_colors:
            bakeVertexColors(obj)
        else:
            bake_object(obj, folderPath, image_res, export_usd)

        obj.hide_render = True
        obj.hide_viewport = True


def set_opacity_threshold_in_usd(usd_file_path: Path, threshold: float = 0.5):
    """
    Post-process USD file to set opacity threshold to 0.5 on all UsdPreviewSurface shaders
    that have alpha/opacity textures connected.
    """
    try:
        from pxr import Sdf, Usd, UsdShade
    except ImportError:
        logging.warning(
            "pxr (USD Python) not available, cannot set opacity threshold in USD file"
        )
        return

    stage = Usd.Stage.Open(str(usd_file_path))
    if not stage:
        logging.warning(f"Could not open USD file: {usd_file_path}")
        return

    # Find all materials and their PreviewSurface shaders
    materials_updated = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            material = UsdShade.Material(prim)
            # Get the surface output to find the shader
            surface_output = material.GetSurfaceOutput()
            if surface_output:
                # Find connected shader
                shader_source = surface_output.GetConnectedSource()
                if shader_source:
                    shader_prim = shader_source[0].GetPrim()
                    shader = UsdShade.Shader(shader_prim)

                    # Check if it's a PreviewSurface shader
                    shader_id = shader.GetIdAttr().Get()
                    if shader_id == "UsdPreviewSurface":
                        # Set opacity threshold on ALL PreviewSurface shaders (like roughness threshold)
                        # This ensures opacity threshold is always 0.5 for all materials
                        opacity_threshold_input = shader.GetInput("opacityThreshold")
                        if opacity_threshold_input:
                            opacity_threshold_input.Set(threshold)
                        else:
                            # Create the input if it doesn't exist
                            opacity_threshold_input = shader.CreateInput(
                                "opacityThreshold", Sdf.ValueTypeNames.Float
                            )
                            opacity_threshold_input.Set(threshold)
                        materials_updated += 1
                        logging.info(
                            f"Set opacity threshold to {threshold} for PreviewSurface shader at {shader_prim.GetPath()}"
                        )

    if materials_updated > 0:
        stage.Save()
        logging.info(
            f"Updated opacity threshold for {materials_updated} materials in {usd_file_path}"
        )
    else:
        logging.info(f"No materials with opacity textures found in {usd_file_path}")


def run_blender_export(
    exportPath: Path, format: str, vertex_colors: bool, individual_export: bool
):
    assert exportPath.parent.exists()
    exportPath = str(exportPath)

    if format == "obj":
        if vertex_colors:
            bpy.ops.wm.obj_export(
                filepath=exportPath,
                export_colors=True,
                export_eval_mode="DAG_EVAL_RENDER",
                export_selected_objects=individual_export,
            )
        else:
            bpy.ops.wm.obj_export(
                filepath=exportPath,
                path_mode="COPY",
                export_materials=True,
                export_pbr_extensions=True,
                export_eval_mode="DAG_EVAL_RENDER",
                export_selected_objects=individual_export,
            )

    if format == "fbx":
        if vertex_colors:
            bpy.ops.export_scene.fbx(
                filepath=exportPath, colors_type="SRGB", use_selection=individual_export
            )
        else:
            bpy.ops.export_scene.fbx(
                filepath=exportPath,
                path_mode="COPY",
                embed_textures=True,
                use_selection=individual_export,
            )

    if format == "stl":
        bpy.ops.export_mesh.stl(filepath=exportPath, use_selection=individual_export)

    if format == "ply":
        bpy.ops.wm.ply_export(
            filepath=exportPath, export_selected_objects=individual_export
        )

    if format in ["usda", "usdc"]:
        bpy.ops.wm.usd_export(
            filepath=exportPath,
            export_textures=True,
            # use_instancing=True,
            overwrite_textures=True,
            selected_objects_only=individual_export,
            root_prim_path="/World",
        )
        # Post-process USD file to set opacity threshold to 0.5 for materials with alpha
        if Path(exportPath).exists():
            try:
                set_opacity_threshold_in_usd(exportPath, threshold=0.5)
            except Exception as e:
                logging.warning(f"Could not set opacity threshold in USD file: {e}")


def export_scene(
    input_blend: Path,
    output_folder: Path,
    pipeline_folder=None,
    task_uniqname=None,
    **kwargs,
):
    folder = output_folder / f"export_{input_blend.name}"
    folder.mkdir(exist_ok=True, parents=True)
    export_curr_scene(folder, **kwargs)

    if pipeline_folder is not None and task_uniqname is not None:
        (pipeline_folder / "logs" / f"FINISH_{task_uniqname}").touch()

    return folder


# side effects: will remove parents of inputted obj and clean its name, hides viewport of all objects
def export_single_obj(
    obj: bpy.types.Object,
    output_folder: Path,
    format="usdc",
    image_res=1024,
    vertex_colors=False,
):
    export_usd = format in ["usda", "usdc"]

    export_folder = output_folder
    export_folder.mkdir(parents=True, exist_ok=True)
    export_file = export_folder / output_folder.with_suffix(f".{format}").name

    logging.info(f"Exporting to directory {export_folder=}")

    remove_obj_parents(obj)
    rename_all_meshes(obj)

    collection_views, obj_views = update_visibility()

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 1  # choose render sample
    # Set the tile size
    bpy.context.scene.cycles.tile_x = image_res
    bpy.context.scene.cycles.tile_y = image_res

    if obj.type != "MESH" or obj not in list(bpy.context.view_layer.objects):
        raise ValueError("Object not mesh")

    if export_usd:
        apply_all_modifiers(obj)
    else:
        realizeInstances(obj)
        apply_all_modifiers(obj)

    if not skipBake(obj) and format != "stl":
        if vertex_colors:
            bakeVertexColors(obj)
        else:
            obj.hide_render = False
            obj.hide_viewport = False
            bake_object(obj, export_folder / "textures", image_res, export_usd)
            obj.hide_render = True
            obj.hide_viewport = True

    for collection, status in collection_views.items():
        collection.hide_render = status

    for obj, status in obj_views.items():
        obj.hide_render = status

    clean_names(obj)

    old_loc = obj.location.copy()
    obj.location = (0, 0, 0)

    if (
        obj.type != "MESH"
        or obj.hide_render
        or len(obj.data.vertices) == 0
        or obj not in list(bpy.context.view_layer.objects)
    ):
        raise ValueError("Object is not mesh or hidden from render")

    export_subfolder = export_folder / obj.name
    export_subfolder.mkdir(exist_ok=True)
    export_file = export_subfolder / f"{obj.name}.{format}"

    logging.info(f"Exporting file to {export_file=}")
    obj.hide_viewport = False
    obj.select_set(True)
    run_blender_export(export_file, format, vertex_colors, individual_export=True)
    obj.select_set(False)
    obj.location = old_loc

    return export_file


def export_sim_ready(
    obj: bpy.types.Object,
    output_folder: Path,
    image_res: int = 1024,
    translation: Tuple = (0, 0, 0),
    name: Optional[str] = None,
    visual_only: bool = False,
    collision_only: bool = False,
    separate_asset_dirs: bool = True,
) -> Dict[str, List[Path]]:
    """
    Exports both the visual and collision assets for a geometry.
    """
    if not visual_only:
        assert coacd is not None, "coacd is required to export simulation assets."

    asset_exports = defaultdict(list)
    export_name = name if name is not None else obj.name

    if separate_asset_dirs:
        visual_export_folder = output_folder / "visual"
        collision_export_folder = output_folder / "collision"
    else:
        visual_export_folder = output_folder
        collision_export_folder = output_folder

    texture_export_folder = output_folder / "textures"

    visual_export_folder.mkdir(parents=True, exist_ok=True)
    collision_export_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Exporting to directory {output_folder=}")

    collection_views, obj_views = update_visibility()

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 1  # choose render sample
    # Set the tile size
    bpy.context.scene.cycles.tile_x = image_res
    bpy.context.scene.cycles.tile_y = image_res

    if obj.type != "MESH" or obj not in list(bpy.context.view_layer.objects):
        raise ValueError("Object not mesh")

    # export the textures
    if not skipBake(obj):
        texture_export_folder.mkdir(parents=True, exist_ok=True)
        obj.hide_render = False
        obj.hide_viewport = False
        bake_object(obj, texture_export_folder, image_res, False, export_name)
        obj.hide_render = True
        obj.hide_viewport = True

    for collection, status in collection_views.items():
        collection.hide_render = status

    for obj_tmp, status in obj_views.items():
        obj_tmp.hide_render = status

    # translating object
    old_loc = obj.location.copy()
    obj.location = (
        old_loc[0] + translation[0],
        old_loc[1] + translation[1],
        old_loc[2] + translation[2],
    )

    if (
        obj.type != "MESH"
        or obj.hide_render
        or len(obj.data.vertices) == 0
        or obj not in list(bpy.context.view_layer.objects)
    ):
        raise ValueError("Object is not mesh or hidden from render")

    # export the mesh assets
    visual_export_file = visual_export_folder / f"{export_name}.obj"

    logging.info(f"Exporting file to {visual_export_file=}")
    obj.hide_viewport = False
    obj.select_set(True)

    # export visual asset
    with butil.SelectObjects(obj, active=1):
        bpy.ops.wm.obj_export(
            filepath=str(visual_export_file),
            up_axis="Z",
            forward_axis="Y",
            export_selected_objects=True,
            export_triangulated_mesh=True,  # required for coacd to run properly
        )
    if not collision_only:
        asset_exports["visual"].append(visual_export_file)

    if visual_only:
        obj.select_set(False)
        obj.location = old_loc
        return asset_exports

    clone = butil.deep_clone_obj(obj)
    parts = butil.split_object(clone)

    part_export_obj_file = visual_export_folder / f"{export_name}_part.obj"
    part_export_mtl_file = visual_export_folder / f"{export_name}_part.mtl"

    collision_count = 0
    for part in parts:
        with butil.SelectObjects(part, active=1):
            bpy.ops.wm.obj_export(
                filepath=str(part_export_obj_file),
                up_axis="Z",
                forward_axis="Y",
                export_selected_objects=True,
                export_triangulated_mesh=True,  # required for coacd to run properly
            )

        # export the collision meshes
        mesh_tri = trimesh.load(
            str(part_export_obj_file), merge_norm=True, merge_tex=True, force="mesh"
        )
        trimesh.repair.fix_inversion(mesh_tri)
        preprocess_mode = "off"
        if not mesh_tri.is_volume:
            print(
                mesh_tri.is_watertight,
                mesh_tri.is_winding_consistent,
                np.isfinite(mesh_tri.center_mass).all(),
                mesh_tri.volume > 0.0,
            )
            preprocess_mode = "on"

            if len(mesh_tri.vertices) < 4:
                logging.warning(
                    f"Mesh is not a volume. Only has {len(mesh_tri.vertices)} vertices."
                )
                # raise ValueError(f"Mesh is not a volume. Only has {len(mesh_tri.vertices)} vertices.")
        mesh = coacd.Mesh(mesh_tri.vertices, mesh_tri.faces)

        subparts = coacd.run_coacd(
            mesh=mesh,
            threshold=0.05,
            max_convex_hull=-1,
            preprocess_mode=preprocess_mode,
            mcts_max_depth=3,
        )
        export_name = export_name.replace("vis", "col")
        for vs, fs in subparts:
            collision_export_file = (
                collision_export_folder / f"{export_name}_col{collision_count}.obj"
            )
            subpart_mesh = trimesh.Trimesh(vs, fs)

            # if subpart_mesh.is_empty:
            #     raise ValueError(
            #         "Warning: Collision mesh is completely outside the bounds of the original mesh."
            #     )
            subpart_mesh.export(str(collision_export_file))
            asset_exports["collision"].append(collision_export_file)
            collision_count += 1

    # delete temporary part files
    part_export_obj_file.unlink(missing_ok=True)
    part_export_mtl_file.unlink(missing_ok=True)

    obj.select_set(False)
    obj.location = old_loc
    butil.delete(clone)

    return asset_exports


@gin.configurable
def export_curr_scene(
    output_folder: Path,
    format="usdc",
    image_res=1024,
    vertex_colors=False,
    individual_export=False,
    omniverse_export=False,
    pipeline_folder=None,
    task_uniqname=None,
) -> Path:
    export_usd = format in ["usda", "usdc"]

    export_folder = output_folder
    export_folder.mkdir(exist_ok=True)
    export_file = export_folder / output_folder.with_suffix(f".{format}").name

    logging.info(f"Exporting to directory {export_folder=}")

    remove_obj_parents()
    delete_objects()
    triangulate_meshes()
    if omniverse_export:
        split_glass_mats()
    rename_all_meshes()

    scatter_cols = []
    if export_usd:
        if bpy.data.collections.get("scatter"):
            scatter_cols.append(bpy.data.collections["scatter"])
        if bpy.data.collections.get("scatters"):
            scatter_cols.append(bpy.data.collections["scatters"])
        for col in scatter_cols:
            for obj in col.all_objects:
                remove_shade_smooth(obj)

    # remove 0 polygon meshes except for scatters
    # if export_usd:
    #     for obj in bpy.data.objects:
    #         if obj.type == 'MESH' and len(obj.data.polygons) == 0:
    #             if scatter_cols is not None:
    #                 if any(x in scatter_cols for x in obj.users_collection):
    #                      continue
    #             logging.info(f"{obj.name} has no faces, removing...")
    #             bpy.data.objects.remove(obj, do_unlink=True)

    collection_views, obj_views = update_visibility()

    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj not in list(bpy.context.view_layer.objects):
            continue
        if export_usd:
            apply_all_modifiers(obj)
        else:
            realizeInstances(obj)
            apply_all_modifiers(obj)

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 1  # choose render sample
    # Set the tile size
    bpy.context.scene.cycles.tile_x = image_res
    bpy.context.scene.cycles.tile_y = image_res

    # iterate through all objects and bake them
    bake_scene(
        folderPath=export_folder / "textures",
        image_res=image_res,
        vertex_colors=vertex_colors,
        export_usd=export_usd,
    )

    for collection, status in collection_views.items():
        collection.hide_render = status

    for obj, status in obj_views.items():
        obj.hide_render = status

    clean_names()

    for obj in bpy.data.objects:
        obj.hide_viewport = obj.hide_render

    if omniverse_export:
        adjust_wattages()
        set_center_of_mass()
        # remove 0 polygon meshes
        for obj in bpy.data.objects:
            if obj.type == "MESH" and len(obj.data.polygons) == 0:
                logging.info(f"{obj.name} has no faces, removing...")
                bpy.data.objects.remove(obj, do_unlink=True)

    if individual_export:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.location_clear()  # send all objects to (0,0,0)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.data.objects:
            if (
                obj.type != "MESH"
                or obj.hide_render
                or len(obj.data.vertices) == 0
                or obj not in list(bpy.context.view_layer.objects)
            ):
                continue

            obj_name = obj.name.replace("/", "_")
            export_subfolder = export_folder / obj_name
            export_subfolder.mkdir(exist_ok=True, parents=True)
            export_file = export_subfolder / f"{obj_name}.{format}"

            logging.info(f"Exporting file to {export_file=}")
            obj.hide_viewport = False
            obj.select_set(True)
            run_blender_export(export_file, format, vertex_colors, individual_export)
            obj.select_set(False)
    else:
        logging.info(f"Exporting file to {export_file=}")
        run_blender_export(export_file, format, vertex_colors, individual_export)

        return export_file


def main(args):
    args.output_folder.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=args.output_folder / "export_logs.log",
        level=logging.DEBUG,
        filemode="w+",
    )

    targets = sorted(list(args.input_folder.iterdir()))
    for blendfile in targets:
        if blendfile.stem == "solve_state":
            shutil.copy(blendfile, args.output_folder / "solve_state.json")

        if not blendfile.suffix == ".blend":
            print(f"Skipping non-blend file {blendfile}")
            continue

        bpy.ops.wm.open_mainfile(filepath=str(blendfile))

        folder = export_scene(
            blendfile,
            args.output_folder,
            format=args.format,
            image_res=args.resolution,
            vertex_colors=args.vertex_colors,
            individual_export=args.individual,
            omniverse_export=args.omniverse,
        )
        # wanted to use shutil here but kept making corrupted files
        subprocess.call(["zip", "-r", str(folder.with_suffix(".zip")), str(folder)])

    bpy.ops.wm.quit_blender()


def make_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_folder", type=Path)
    parser.add_argument("--output_folder", type=Path)

    parser.add_argument("-f", "--format", type=str, choices=FORMAT_CHOICES)

    parser.add_argument("-v", "--vertex_colors", action="store_true")
    parser.add_argument("-r", "--resolution", default=1024, type=int)
    parser.add_argument("-i", "--individual", action="store_true")
    parser.add_argument("-o", "--omniverse", action="store_true")

    args = parser.parse_args()

    if args.format not in FORMAT_CHOICES:
        raise ValueError("Unsupported or invalid file format.")

    if args.vertex_colors and args.format not in ["ply", "fbx", "obj"]:
        raise ValueError("File format does not support vertex colors.")

    if args.format == "ply" and not args.vertex_colors:
        raise ValueError(".ply export must use vertex colors.")

    return args


if __name__ == "__main__":
    args = make_args()
    main(args)
