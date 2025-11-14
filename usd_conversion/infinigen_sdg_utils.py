# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
import os
import random
import re
from itertools import chain

from numpy.random import f

import omni.kit.app
import omni.kit.commands
import omni.physx
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.semantics import add_update_semantics, remove_all_semantics
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
import isaacsim.core.utils.prims as prim_utils

# Seeding for reproducibility (set in infinigen_sdg.py if debug_mode is True then seed is 10)
# random.seed(42)
# rep.set_global_seed(42)


def set_transform_attributes(
    prim: Usd.Prim,
    location: Gf.Vec3d | None = None,
    orientation: Gf.Quatf | None = None,
    rotation: Gf.Vec3f | None = None,
    scale: Gf.Vec3f | None = None,
) -> None:
    """Set transformation attributes (location, orientation, rotation, scale) on a prim."""
    if location is not None:
        if not prim.HasAttribute("xformOp:translate"):
            UsdGeom.Xformable(prim).AddTranslateOp()
        prim.GetAttribute("xformOp:translate").Set(location)
    if orientation is not None:
        if not prim.HasAttribute("xformOp:orient"):
            UsdGeom.Xformable(prim).AddOrientOp()
        prim.GetAttribute("xformOp:orient").Set(orientation)
    if rotation is not None:
        if not prim.HasAttribute("xformOp:rotateXYZ"):
            UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute("xformOp:rotateXYZ").Set(rotation)
    if scale is not None:
        if not prim.HasAttribute("xformOp:scale"):
            UsdGeom.Xformable(prim).AddScaleOp()
        prim.GetAttribute("xformOp:scale").Set(scale)


def add_colliders(root_prim: Usd.Prim, approximation_type: str = "convexHull") -> None:
    """Add collision attributes to mesh and geometry primitives under the root prim."""
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Gprim):
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)

        if desc_prim.IsA(UsdGeom.Mesh):
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set(approximation_type)


def has_colliders(root_prim: Usd.Prim) -> bool:
    """Check if any descendant prims under the root prim have collision attributes."""
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.HasAPI(UsdPhysics.CollisionAPI):
            return True
    return False


def add_rigid_body_dynamics(prim: Usd.Prim, disable_gravity: bool = False) -> None:
    """Add rigid body dynamics properties to a prim if it has colliders, with optional gravity setting."""
    if has_colliders(prim):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        else:
            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)

        # Apply PhysX rigid body dynamics
        if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        else:
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
    else:
        print(
            f"[SDG-Infinigen] Prim '{prim.GetPath()}' has no colliders. Skipping adding rigid body dynamics properties."
        )


def add_colliders_and_rigid_body_dynamics(
    prim: Usd.Prim, disable_gravity: bool = False
) -> None:
    """Add colliders and rigid body dynamics properties to a prim, with optional gravity setting."""
    add_colliders(prim)
    add_rigid_body_dynamics(prim, disable_gravity)


def get_random_pose_on_sphere(
    origin: tuple[float, float, float],
    radius_range: tuple[float, float],
    polar_angle_range: tuple[float, float],
    camera_forward_axis: tuple[float, float, float] = (0, 0, -1),
) -> tuple[Gf.Vec3d, Gf.Quatf]:
    """Generate a random pose on a sphere looking at the origin, with specified radius and polar angle ranges."""
    # https://docs.omniverse.nvidia.com/isaacsim/latest/reference_conventions.html
    # Convert degrees to radians for polar angles (theta)
    polar_angle_min_rad = math.radians(polar_angle_range[0])
    polar_angle_max_rad = math.radians(polar_angle_range[1])

    # Generate random spherical coordinates
    radius = random.uniform(radius_range[0], radius_range[1])
    polar_angle = random.uniform(polar_angle_min_rad, polar_angle_max_rad)
    azimuthal_angle = random.uniform(0, 2 * math.pi)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * math.sin(polar_angle) * math.cos(azimuthal_angle)
    y = radius * math.sin(polar_angle) * math.sin(azimuthal_angle)
    z = radius * math.cos(polar_angle)

    # Calculate the location in 3D space
    location = Gf.Vec3d(origin[0] + x, origin[1] + y, origin[2] + z)

    # Calculate direction vector from camera to look_at point
    direction = Gf.Vec3d(origin) - location
    direction_normalized = direction.GetNormalized()

    # Calculate rotation from forward direction (rotateFrom) to direction vector (rotateTo)
    rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), direction_normalized)
    orientation = Gf.Quatf(rotation.GetQuat())

    return location, orientation


def randomize_camera_poses(
    cameras: list[Usd.Prim],
    targets: list[Usd.Prim],
    distance_range: tuple[float, float],
    polar_angle_range: tuple[float, float] = (0, 180),
    look_at_offset: tuple[float, float] = (-0.1, 0.1),
) -> None:
    """Randomize the poses of cameras to look at random targets with adjustable distance and offset."""
    for cam in cameras:
        # Get a random target asset to look at
        target_asset = random.choice(targets)

        # Add a look_at offset so the target is not always in the center of the camera view
        target_loc = target_asset.GetAttribute("xformOp:translate").Get()
        target_loc = (
            target_loc[0] + random.uniform(look_at_offset[0], look_at_offset[1]),
            target_loc[1] + random.uniform(look_at_offset[0], look_at_offset[1]),
            target_loc[2] + random.uniform(look_at_offset[0], look_at_offset[1]),
        )

        # Generate random camera pose
        loc, quat = get_random_pose_on_sphere(
            target_loc, distance_range, polar_angle_range
        )

        # Set the camera's transform attributes to the generated location and orientation
        set_transform_attributes(cam, location=loc, orientation=quat)


def get_usd_paths_from_folder(
    folder_path: str,
    recursive: bool = True,
    usd_paths: list[str] = None,
    skip_keywords: list[str] = None,
) -> list[str]:
    """Retrieve USD file paths from a folder, optionally searching recursively and filtering by keywords."""
    if usd_paths is None:
        usd_paths = []
    skip_keywords = skip_keywords or []

    # Make sure the omni.client extension is enabled
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.client"):
        ext_manager.set_extension_enabled_immediate("omni.client", True)
    import omni.client

    result, entries = omni.client.list(folder_path)
    if result != omni.client.Result.OK:
        print(f"[SDG-Infinigen] Could not list assets in path: {folder_path}")
        return usd_paths

    for entry in entries:
        if any(
            keyword.lower() in entry.relative_path.lower() for keyword in skip_keywords
        ):
            continue
        _, ext = os.path.splitext(entry.relative_path)
        if ext in [".usd", ".usda", ".usdc"]:
            path_posix = os.path.join(folder_path, entry.relative_path).replace(
                "\\", "/"
            )
            usd_paths.append(path_posix)
        elif recursive and entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            sub_folder = os.path.join(folder_path, entry.relative_path).replace(
                "\\", "/"
            )
            get_usd_paths_from_folder(
                sub_folder,
                recursive=recursive,
                usd_paths=usd_paths,
                skip_keywords=skip_keywords,
            )

    return usd_paths


def get_usd_paths(
    files: list[str] = None,
    folders: list[str] = None,
    skip_folder_keywords: list[str] = None,
) -> list[str]:
    """Retrieve USD paths from specified files and folders, optionally filtering out specific folder keywords."""
    files = files or []
    folders = folders or []
    skip_folder_keywords = skip_folder_keywords or []

    assets_root_path = get_assets_root_path()
    env_paths = []

    for file_path in files:
        file_path = (
            file_path
            if file_path.startswith(("omniverse://", "http://", "https://", "file://"))
            else assets_root_path + file_path
        )
        env_paths.append(file_path)

    for folder_path in folders:
        folder_path = (
            folder_path
            if folder_path.startswith(
                ("omniverse://", "http://", "https://", "file://")
            )
            else assets_root_path + folder_path
        )
        env_paths.extend(
            get_usd_paths_from_folder(
                folder_path, recursive=True, skip_keywords=skip_folder_keywords
            )
        )

    return env_paths


def load_env(usd_path: str, prim_path: str, remove_existing: bool = True) -> Usd.Prim:
    """Load an environment from a USD file into the stage at the specified prim path, optionally removing any existing prim."""
    stage = omni.usd.get_context().get_stage()

    # Remove existing prim if specified
    if remove_existing and stage.GetPrimAtPath(prim_path):
        omni.kit.commands.execute("DeletePrimsCommand", paths=[prim_path])

    root_prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    return root_prim


def add_colliders_to_env(
    root_path: str | None = None, approximation_type: str = "none"
) -> None:
    """Add colliders to all mesh prims within the specified root path in the stage."""
    stage = omni.usd.get_context().get_stage()
    prim = (
        stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)
    )

    for prim in Usd.PrimRange(prim):
        if prim.IsA(UsdGeom.Mesh):
            add_colliders(prim, approximation_type)


def find_matching_prims(
    match_strings: list[str],
    root_path: str | None = None,
    prim_type: str | None = None,
    first_match_only: bool = False,
) -> Usd.Prim | list[Usd.Prim] | None:
    """Find prims matching specified strings, with optional type filtering and single match return."""
    stage = omni.usd.get_context().get_stage()
    root_prim = (
        stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)
    )

    matching_prims = []
    for prim in Usd.PrimRange(root_prim):
        if any(match in str(prim.GetPath()) for match in match_strings):
            if prim_type is None or prim.GetTypeName() == prim_type:
                if first_match_only:
                    return prim
                matching_prims.append(prim)

    return matching_prims if not first_match_only else None


def hide_matching_prims(
    match_strings: list[str], root_path: str | None = None, prim_type: str | None = None
) -> None:
    """Set visibility of prims matching specified strings to 'invisible' within the root path."""
    stage = omni.usd.get_context().get_stage()
    root_prim = (
        stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)
    )

    for prim in Usd.PrimRange(root_prim):
        if prim_type is None or prim.GetTypeName() == prim_type:
            if any(match in str(prim.GetPath()) for match in match_strings):
                prim.GetAttribute("visibility").Set("invisible")


def setup_env(
    root_path: str | None = None,
    approximation_type: str = "none",
    hide_top_walls: bool = False,
) -> None:
    """Set up the environment with colliders, ceiling light adjustments, and optional top wall hiding."""
    # Fix ceiling lights: meshes are blocking the light and need to be set to invisible
    ceiling_light_meshes = find_matching_prims(["001_SPLIT_GLA"], root_path, "Xform")
    for light_mesh in ceiling_light_meshes:
        light_mesh.GetAttribute("visibility").Set("invisible")

    # Hide ceiling light meshes for lighting fix
    hide_matching_prims(["001_SPLIT_GLA"], root_path, "Xform")

    """# Hide top walls for better debug view, if specified
    if hide_top_walls:
        hide_matching_prims(["_exterior", "_ceiling"], root_path)"""

    # Add colliders to the environment
    add_colliders_to_env(root_path, approximation_type)

    # Fix dining table collision by setting it to a bounding cube approximation
    table_prim = find_matching_prims(
        match_strings=["TableDining"],
        root_path=root_path,
        prim_type="Xform",
        first_match_only=True,
    )
    if table_prim is not None:
        add_colliders(table_prim, approximation_type="boundingCube")
    else:
        print(
            "[SDG-Infinigen] Could not find dining table prim in the environment which is ok, it's not needed."
        )


def create_shape_distractors(
    num_distractors: int,
    shape_types: list[str],
    root_path: str,
    gravity_disabled_chance: float,
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Create shape distractors with optional gravity settings, returning lists of floating and falling shapes."""
    stage = omni.usd.get_context().get_stage()
    floating_shapes = []
    falling_shapes = []
    for _ in range(num_distractors):
        rand_shape = random.choice(shape_types)
        disable_gravity = random.random() < gravity_disabled_chance
        name_prefix = "floating_" if disable_gravity else "falling_"
        prim_path = omni.usd.get_stage_next_free_path(
            stage, f"{root_path}/{name_prefix}{rand_shape}", False
        )
        prim = stage.DefinePrim(prim_path, rand_shape.capitalize())
        add_colliders_and_rigid_body_dynamics(prim, disable_gravity=disable_gravity)
        (floating_shapes if disable_gravity else falling_shapes).append(prim)
    return floating_shapes, falling_shapes


def load_shape_distractors(
    shape_distractors_config: dict,
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Load shape distractors based on configuration, returning lists of floating and falling shapes."""
    num_shapes = shape_distractors_config.get("num", 0)
    shape_types = shape_distractors_config.get(
        "shape_types", ["capsule", "cone", "cylinder", "sphere", "cube"]
    )
    shape_gravity_disabled_chance = shape_distractors_config.get(
        "gravity_disabled_chance", 0.0
    )
    return create_shape_distractors(
        num_shapes, shape_types, "/Distractors", shape_gravity_disabled_chance
    )


def create_mesh_distractors(
    num_distractors: int,
    mesh_urls: list[str],
    root_path: str,
    gravity_disabled_chance: float,
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Create mesh distractors from specified URLs with optional gravity settings."""
    stage = omni.usd.get_context().get_stage()
    floating_meshes = []
    falling_meshes = []
    for _ in range(num_distractors):
        rand_mesh_url = random.choice(mesh_urls)
        disable_gravity = random.random() < gravity_disabled_chance
        name_prefix = "floating_" if disable_gravity else "falling_"
        prim_name = os.path.basename(rand_mesh_url).split(".")[0]
        prim_path = omni.usd.get_stage_next_free_path(
            stage, f"{root_path}/{name_prefix}{prim_name}", False
        )
        try:
            prim = add_reference_to_stage(usd_path=rand_mesh_url, prim_path=prim_path)
        except Exception as e:
            print(
                f"[SDG-Infinigen] Failed to load mesh distractor reference {rand_mesh_url} with exception: {e}"
            )
            continue
        add_colliders_and_rigid_body_dynamics(prim, disable_gravity=disable_gravity)
        (floating_meshes if disable_gravity else falling_meshes).append(prim)
    return floating_meshes, falling_meshes


def load_mesh_distractors(
    mesh_distractors_config: dict,
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Load mesh distractors based on configuration, returning lists of floating and falling meshes."""
    num_meshes = mesh_distractors_config.get("num", 0)
    mesh_gravity_disabled_chance = mesh_distractors_config.get(
        "gravity_disabled_chance", 0.0
    )
    mesh_folders = mesh_distractors_config.get("folders", [])
    mesh_files = mesh_distractors_config.get("files", [])
    mesh_urls = get_usd_paths(
        files=mesh_files,
        folders=mesh_folders,
        skip_folder_keywords=["material", "texture", ".thumbs"],
    )
    floating_meshes, falling_meshes = create_mesh_distractors(
        num_meshes, mesh_urls, "/Distractors", mesh_gravity_disabled_chance
    )
    for prim in chain(floating_meshes, falling_meshes):
        remove_all_semantics(prim, recursive=True)
    return floating_meshes, falling_meshes


def create_auto_labeled_assets(
    num_assets: int,
    asset_urls: list[str],
    root_path: str,
    regex_replace_pattern: str,
    regex_replace_repl: str,
    gravity_disabled_chance: float,
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Create assets with automatic labels, applying optional gravity settings."""
    stage = omni.usd.get_context().get_stage()
    floating_assets = []
    falling_assets = []
    for _ in range(num_assets):
        asset_url = random.choice(asset_urls)
        disable_gravity = random.random() < gravity_disabled_chance
        name_prefix = "floating_" if disable_gravity else "falling_"
        basename = os.path.basename(asset_url)
        name_without_ext = os.path.splitext(basename)[0]
        label = re.sub(regex_replace_pattern, regex_replace_repl, name_without_ext)
        prim_path = omni.usd.get_stage_next_free_path(
            stage, f"{root_path}/{name_prefix}{label}", False
        )
        try:
            prim = add_reference_to_stage(usd_path=asset_url, prim_path=prim_path)
        except Exception as e:
            print(
                f"[SDG-Infinigen] Failed to load mesh distractor reference {asset_url} with exception: {e}"
            )
            continue
        add_colliders_and_rigid_body_dynamics(prim, disable_gravity=disable_gravity)
        remove_all_semantics(prim, recursive=True)
        add_update_semantics(prim, label)
        (floating_assets if disable_gravity else falling_assets).append(prim)
    return floating_assets, falling_assets


def load_auto_labeled_assets(
    auto_label_config: dict,
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Load auto-labeled assets based on configuration, returning lists of floating and falling assets."""
    num_assets = auto_label_config.get("num", 0)
    gravity_disabled_chance = auto_label_config.get("gravity_disabled_chance", 0.0)
    assets_files = auto_label_config.get("files", [])
    assets_folders = auto_label_config.get("folders", [])
    assets_urls = get_usd_paths(
        files=assets_files,
        folders=assets_folders,
        skip_folder_keywords=["material", "texture", ".thumbs"],
    )
    regex_replace_pattern = auto_label_config.get("regex_replace_pattern", "")
    regex_replace_repl = auto_label_config.get("regex_replace_repl", "")
    return create_auto_labeled_assets(
        num_assets,
        assets_urls,
        "/Assets",
        regex_replace_pattern,
        regex_replace_repl,
        gravity_disabled_chance,
    )


def create_labeled_assets(
    num_assets: int,
    asset_url: str,
    label: str,
    root_path: str,
    gravity_disabled_chance: float,
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Create labeled assets with optional gravity settings, returning lists of floating and falling assets."""
    stage = omni.usd.get_context().get_stage()
    assets_root_path = get_assets_root_path()
    asset_url = (
        asset_url
        if asset_url.startswith(("omniverse://", "http://", "https://", "file://"))
        else assets_root_path + asset_url
    )
    floating_assets = []
    falling_assets = []
    for _ in range(num_assets):
        disable_gravity = random.random() < gravity_disabled_chance
        name_prefix = "floating_" if disable_gravity else "falling_"
        prim_path = omni.usd.get_stage_next_free_path(
            stage, f"{root_path}/{name_prefix}{label}", False
        )

        prim = add_reference_to_stage(usd_path=asset_url, prim_path=prim_path)
        add_colliders_and_rigid_body_dynamics(prim, disable_gravity=disable_gravity)
        remove_all_semantics(prim, recursive=True)
        add_update_semantics(prim, label)
        (floating_assets if disable_gravity else falling_assets).append(prim)
    return floating_assets, falling_assets


def load_manual_labeled_assets(
    manual_labeled_assets_config: list[dict],
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Load manually labeled assets based on configuration, returning lists of floating and falling assets."""
    labeled_floating_assets = []
    labeled_falling_assets = []
    for labeled_asset_config in manual_labeled_assets_config:
        asset_url = labeled_asset_config.get("url", "")
        asset_label = labeled_asset_config.get("label", "")
        num_assets = labeled_asset_config.get("num", 0)
        gravity_disabled_chance = labeled_asset_config.get(
            "gravity_disabled_chance", 0.0
        )
        floating_assets, falling_assets = create_labeled_assets(
            num_assets,
            asset_url,
            asset_label,
            "/Assets",
            gravity_disabled_chance,
        )
        labeled_floating_assets.extend(floating_assets)
        labeled_falling_assets.extend(falling_assets)
    return labeled_floating_assets, labeled_falling_assets


## CUSTOM TODO HERE:
## - Add a way to specify the plane
def create_planes_with_defect_materials(
    num_planes: int,
    material_paths: list[str],
    root_path: str,
    gravity_disabled_chance: float,
    plane_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Create planes with defect materials as labeled assets."""
    stage = omni.usd.get_context().get_stage()
    floating_planes = []
    falling_planes = []

    # First, load the defect materials into the stage
    load_defect_materials_to_stage()

    wall_defects = []

    for _ in range(num_planes):
        disable_gravity = random.random() < gravity_disabled_chance

        wall_defect = rep.create.plane(
            position=(0, 0, 0),
            scale=plane_scale,
            semantics=[("class", "defect")],
        )

        # Add physics properties
        """add_colliders_and_rigid_body_dynamics(
            wall_defect, disable_gravity=disable_gravity
        )"""

        # Add semantics
        """remove_all_semantics(defect_prim, recursive=True)
        add_update_semantics(defect_prim, "defect")"""

        # Apply random defect material
        with wall_defect:
            rep.randomizer.materials(material_paths)

        wall_defects.append(wall_defect)

    # Find the created prims
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xform):
            if prim.GetName().startswith("Plane"):
                floating_planes.append(prim)

    # (floating_planes if disable_gravity else falling_planes).append(wall_defect)

    return floating_planes, falling_planes, wall_defects


def get_planes_with_defect_materials() -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Create planes with defect materials as labeled assets."""
    stage = omni.usd.get_context().get_stage()
    floating_planes = []

    # First, load the defect materials into the stage
    # load_defect_materials_to_stage()
    wall_defects_mesh = []
    # Find the prims
    for prim in stage.Traverse():
        if prim.GetName().startswith("StaticCategoryFactory"):
            if prim.IsA(UsdGeom.Mesh):
                wall_defects_mesh.append(prim)
                set_transform_attributes(prim)
                # add semantics
                rep.modify.semantics(semantics=[("class", "defect")], input_prims=prim)
            if prim.IsA(UsdGeom.Xform):
                floating_planes.append(prim)

    return floating_planes, wall_defects_mesh


def register_lights_addition(root_path: str):
    with rep.trigger.on_custom_event(event_name="lights_addition"):
        ceiling_light_meshes = find_matching_prims(
            ["001_SPLIT_GLA"], root_path, "Xform"
        )
        for light_mesh in ceiling_light_meshes:
            # Get the current position of the light mesh
            mesh_position = light_mesh.GetAttribute("xformOp:translate").Get()
            # Position light 0.5 units below the mesh in Z-axis, same X and Y
            light_position = (
                mesh_position[0],
                mesh_position[1],
                mesh_position[2] - 0.05,
            )

            light = rep.create.light(
                light_type="disk",
                position=light_position,
                rotation=(0, 0, 0),
                scale=(0.5, 0.5, 0.5),
                intensity=rep.distribution.uniform(20000, 20000),
                temperature=rep.distribution.uniform(10000, 10000),
            )

        """Set up the environment with colliders, ceiling light adjustments, and optional top wall hiding."""
    # Fix ceiling lights: meshes are blocking the light and need to be set to invisible
    """ceiling_light_meshes = find_matching_prims(["001_SPLIT_GLA"], root_path, "Xform")
    for light_mesh in ceiling_light_meshes:
        light_mesh.GetAttribute("visibility").Set("invisible")

    # Hide ceiling light meshes for lighting fix
    hide_matching_prims(["001_SPLIT_GLA"], root_path, "Xform")"""
    """light_1 = prim_utils.create_prim(
        "/World/Light_1",
        "SphereLight",
        position=np.array([1.0, 1.0, 1.0]),
        attributes={
            "inputs:radius": 0.01,
            "inputs:intensity": 5e3,
            "inputs:color": (1.0, 0.0, 1.0),
        },
    )"""


def register_randomize_wall_defects_textures(
    wall_defects_mesh: list[Usd.Prim], mdl_paths: list[str]
):
    with rep.trigger.on_custom_event(event_name="randomize_defect_texture"):
        """Randomize the textures of the wall defects meshes."""
        for prim in wall_defects_mesh:
            rep.randomizer.materials(materials=mdl_paths, input_prims=prim)
            rep.modify.pose(
                scale=rep.distribution.normal(0.7, 0.2),
                input_prims=prim,
            )


## CUSTOM DONE HERE
def load_defect_materials_to_stage():
    """Load defect MDL materials into the stage."""
    import os
    import omni.kit.commands

    # Get the path to the defects folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    defects_folder = os.path.join(script_dir, "usds", "mdl", "defects")

    # Create the Looks prim if it doesn't exist
    stage = omni.usd.get_context().get_stage()
    looks_prim = stage.GetPrimAtPath("/World/Looks")
    if not looks_prim:
        looks_prim = stage.DefinePrim("/World/Looks", "Xform")

    # Find all MDL files in the defects folder
    import glob

    mdl_pattern = os.path.join(defects_folder, "*.mdl")
    mdl_files = glob.glob(mdl_pattern)

    if not mdl_files:
        print(f"[SDG-Infinigen] No MDL files found in {defects_folder}")
        return

    # Load each defect MDL file
    for defect_path in mdl_files:
        defect_filename = os.path.basename(defect_path)
        defect_name = os.path.splitext(defect_filename)[0]  # Remove .mdl extension

        material_prim_path = f"/World/Looks/{defect_name}"

        # Load the MDL material
        try:
            omni.kit.commands.execute(
                "CreateMdlMaterialPrim",
                mtl_url=f"file://{defect_path}",
                mtl_name=defect_name,
                mtl_path=material_prim_path,
            )
            print(f"[SDG-Infinigen] Loaded defect material: {defect_filename}")
        except Exception as e:
            print(
                f"[SDG-Infinigen] Failed to load defect material {defect_filename}: {e}"
            )

        # fix UV
        try:
            # Create the primvars:doNotCastShadows attribute
            # Note: primvars are typically created on the prim, not the mesh schema
            prim = stage.GetPrimAtPath(material_prim_path + "/Shader")
            # Check if the attribute already exists
            if prim.HasAttribute("inputs:project_uvw"):
                print(
                    f"    Updating existing inputs:project_uvw attribute for {prim.GetName()}"
                )
                prim.GetAttribute("inputs:project_uvw").Set(True)
            else:
                print(
                    f"    Creating new inputs:project_uvw attribute for {prim.GetName()}"
                )
                # Create the attribute as a boolean
                project_uvw_attr = prim.CreateAttribute(
                    "inputs:project_uvw", Sdf.ValueTypeNames.Bool
                )
                # Set it to False (so shadows WILL be cast)
                project_uvw_attr.Set(True)

            print(f"    Successfully set inputs:project_uvw=True for {prim.GetName()}")

        except Exception as e:
            print(
                f"    ERROR: Failed to set inputs:project_uvw for {prim.GetName()}: {e}"
            )
        try:
            # Create the primvars:doNotCastShadows attribute
            # Note: primvars are typically created on the prim, not the mesh schema
            prim = stage.GetPrimAtPath(material_prim_path + "/Shader")
            # Check if the attribute already exists
            if prim.HasAttribute("inputs:world_or_object"):
                print(
                    f"    Updating existing inputs:world_or_object attribute for {prim.GetName()}"
                )
                prim.GetAttribute("inputs:world_or_object").Set(True)
            else:
                print(
                    f"    Creating new inputs:world_or_object attribute for {prim.GetName()}"
                )
                # Create the attribute as a boolean
                world_or_object_attr = prim.CreateAttribute(
                    "inputs:world_or_object", Sdf.ValueTypeNames.Bool
                )
                # Set it to False (so shadows WILL be cast)
                world_or_object_attr.Set(True)

            print(
                f"    Successfully set inputs:world_or_object=True for {prim.GetName()}"
            )

        except Exception as e:
            print(
                f"    ERROR: Failed to set inputs:world_or_object for {prim.GetName()}: {e}"
            )

        try:
            prim = stage.GetPrimAtPath(material_prim_path + "/Shader")
            # Check if the attribute already exists
            if prim.HasAttribute("inputs:albedo_brightness"):
                print(
                    f"    Updating existing inputs:albedo_brightness attribute for {prim.GetName()}"
                )
                prim.GetAttribute("inputs:albedo_brightness").Set(0.0)
            else:
                print(
                    f"    Creating new inputs:albedo_brightness attribute for {prim.GetName()}"
                )
                # Create the attribute as a boolean
                albedo_brightness_attr = prim.CreateAttribute(
                    "inputs:albedo_brightness", Sdf.ValueTypeNames.Float
                )
                # Set it to False (so shadows WILL be cast)
                albedo_brightness_attr.Set(0.0)

            print(
                f"    Successfully set inputs:albedo_brightness=0.0 for {prim.GetName()}"
            )

        except Exception as e:
            print(
                f"    ERROR: Failed to set inputs:albedo_brightness for {prim.GetName()}: {e}"
            )


## CUSTOM DONE HERE
def get_defect_material_paths():
    """Get paths for defect materials from the usds/mdl/defects/ folder."""
    import os
    import glob

    # Get the path to the defects folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    defects_folder = os.path.join(script_dir, "usds", "mdl", "defects")

    # Find all MDL files in the defects folder
    mdl_pattern = os.path.join(defects_folder, "*.mdl")
    mdl_files = glob.glob(mdl_pattern)

    if not mdl_files:
        print(f"[SDG-Infinigen] No MDL files found in {defects_folder}")
        return []

    # Generate material paths based on found files
    base_path = "/World/Looks/"
    defect_paths = []

    for defect_path in mdl_files:
        defect_filename = os.path.basename(defect_path)
        defect_name = os.path.splitext(defect_filename)[0]  # Remove .mdl extension
        material_path = f"{base_path}{defect_name}"
        defect_paths.append(material_path)

    return defect_paths


def get_cameras_from_stage() -> list[Usd.Prim]:
    """Get the cameras from the stage."""
    stage = omni.usd.get_context().get_stage()
    cameras = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            if prim.GetName().startswith("camera") and prim.GetName().endswith("_0"):
                cameras.append(prim)
                prim.GetAttribute("clippingRange").Set((0.25, 1000))
    return cameras


## CUSTOM DONE HERE
def split_geom_subsets_into_meshes():
    stage = omni.usd.get_context().get_stage()
    windows_meshes = []
    created_meshes = []

    # Find the prims
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            # if prim.GetName().startswith("WindowFactory"):
            if prim:
                windows_meshes.append(prim)

                # Create the mesh schema first, then get geomSubsets
                mesh = UsdGeom.Mesh(prim)
                geom_subsets = UsdGeom.Subset.GetGeomSubsets(mesh)

                if geom_subsets:
                    print(f"Found {len(geom_subsets)} geomSubsets in {prim.GetName()}")

                    # Extract base mesh data with error checking
                    points_attr = mesh.GetPointsAttr()
                    face_vertex_counts_attr = mesh.GetFaceVertexCountsAttr()
                    face_vertex_indices_attr = mesh.GetFaceVertexIndicesAttr()

                    # Check if attributes exist and have values
                    if (
                        not points_attr
                        or not face_vertex_counts_attr
                        or not face_vertex_indices_attr
                    ):
                        print(
                            f"Warning: Missing required attributes in {prim.GetName()}"
                        )
                        continue

                    points = points_attr.Get()
                    face_vertex_counts = face_vertex_counts_attr.Get()
                    face_vertex_indices = face_vertex_indices_attr.Get()

                    # Additional validation
                    if not points or not face_vertex_counts or not face_vertex_indices:
                        print(f"Warning: Empty attribute data in {prim.GetName()}")
                        continue

                    # Get optional attributes
                    normals_attr = mesh.GetNormalsAttr()
                    normals = normals_attr.Get() if normals_attr else None

                    # Process each geomSubset
                    for i, subset in enumerate(geom_subsets):
                        try:
                            # Get subset data with proper error checking
                            indices_attr = subset.GetIndicesAttr()
                            if not indices_attr:
                                print(f"Warning: No indices attribute in subset {i}")
                                continue

                            subset_face_indices = indices_attr.Get()
                            if not subset_face_indices:
                                print(f"Warning: No face indices in subset {i}")
                                continue

                            subset_prim = subset.GetPrim()
                            subset_name = (
                                subset_prim.GetName() if subset_prim else f"subset_{i}"
                            )

                            #########################################################

                            print(
                                f"  Processing subset {i}: {subset_name} with {len(subset_face_indices)} faces"
                            )

                            # Extract subset-specific geometry data
                            subset_face_counts = []
                            subset_face_vertex_indices = []
                            # Filter out face_idx 0 to avoid the problematic case
                            filtered_face_indices = [
                                idx for idx in subset_face_indices if idx != 0
                            ]

                            # Use the filtered indices instead
                            for face_idx in filtered_face_indices:
                                if face_vertex_counts is None:
                                    break

                                if face_idx < len(face_vertex_counts):
                                    face_vertex_count = face_vertex_counts[face_idx]
                                    subset_face_counts.append(face_vertex_count)
                                    # 224 face_vertex_counts

                                    # Get the face vertex indices for this face
                                    face_start_idx = sum(face_vertex_counts[:face_idx])
                                    subset_face_vertex_indices.extend(
                                        face_vertex_indices[
                                            face_start_idx : face_start_idx
                                            + face_vertex_count
                                        ]
                                    )
                                else:
                                    print(
                                        f"    WARNING: Face index {face_idx} is out of bounds for subset {subset_name}"
                                    )

                            # Validate extracted data
                            if not subset_face_counts or not subset_face_vertex_indices:
                                print(
                                    f"Warning: No valid geometry data extracted for subset {subset_name}"
                                )
                                continue

                            # Create new mesh prim path
                            new_mesh_path = f"{prim.GetPath()}_{subset_name}"

                            # Create the new mesh prim
                            new_mesh = UsdGeom.Mesh.Define(stage, new_mesh_path)

                            # Populate the new mesh with extracted data
                            new_mesh.GetPointsAttr().Set(points)
                            new_mesh.GetFaceVertexCountsAttr().Set(subset_face_counts)
                            new_mesh.GetFaceVertexIndicesAttr().Set(
                                subset_face_vertex_indices
                            )

                            # Handle normals if they exist
                            if normals:
                                subset_normals = []
                                for face_idx in subset_face_indices:
                                    if face_idx < len(normals):
                                        subset_normals.append(normals[face_idx])
                                if subset_normals:
                                    new_mesh.GetNormalsAttr().Set(subset_normals)

                            # Copy transform attributes from original mesh
                            copy_transform_attributes(prim, new_mesh)

                            # Copy material binding if it exists - use the subset's prim, not the subset object
                            subset_prim = subset.GetPrim()

                            if subset_prim:
                                if subset_prim.HasAttribute("material:binding"):
                                    material_binding = subset_prim.GetAttribute(
                                        "material:binding"
                                    ).Get()
                                    if material_binding:
                                        # Use the prim object, not the mesh schema for CreateAttribute
                                        new_mesh_prim = new_mesh.GetPrim()
                                        try:
                                            new_attr = new_mesh_prim.CreateAttribute(
                                                "material:binding",
                                                Sdf.ValueTypeNames.Token,
                                            )
                                            new_attr.Set(material_binding)
                                        except Exception as e:
                                            print(
                                                f"    ERROR: Failed to create/set material binding: {e}"
                                            )
                                    else:
                                        print(
                                            f"    DEBUG: Material binding value is None"
                                        )
                                else:
                                    print(
                                        f"    DEBUG: No material:binding attribute found on subset prim"
                                    )

                                    # Check if the original mesh has material binding
                                    if prim.HasAttribute("material:binding"):
                                        original_material = prim.GetAttribute(
                                            "material:binding"
                                        ).Get()

                                        if original_material:
                                            try:
                                                new_mesh_prim = new_mesh.GetPrim()
                                                new_attr = (
                                                    new_mesh_prim.CreateAttribute(
                                                        "material:binding",
                                                        Sdf.ValueTypeNames.Token,
                                                    )
                                                )
                                                new_attr.Set(original_material)
                                            except Exception as e:
                                                print(
                                                    f"    ERROR: Failed to copy original material binding: {e}"
                                                )
                                    else:
                                        print(
                                            f"    DEBUG: Neither subset nor original mesh has material binding"
                                        )
                            else:
                                print(
                                    f"    DEBUG: Subset prim is None, cannot check material binding"
                                )

                            # Also check for other material-related attributes
                            print(
                                f"    DEBUG: Checking for other material attributes on subset prim"
                            )
                            if subset_prim:
                                for attr in subset_prim.GetAttributes():
                                    if "material" in attr.GetName().lower():
                                        print(
                                            f"    DEBUG: Found material-related attribute: {attr.GetName()} = {attr.Get()}"
                                        )

                            # Check original mesh material attributes too
                            print(
                                f"    DEBUG: Checking material attributes on original mesh"
                            )
                            for attr in prim.GetAttributes():
                                if "material" in attr.GetName().lower():
                                    print(
                                        f"    DEBUG: Found material-related attribute on original: {attr.GetName()} = {attr.Get()}"
                                    )

                            created_meshes.append(new_mesh)
                            print(
                                f"Created separate mesh: {new_mesh_path} with {len(subset_face_counts)} faces"
                            )

                            # Check if the newly created mesh should be hidden due to wall_shader
                            new_mesh_name = new_mesh.GetPrim().GetName()
                            if "wall_shader" in new_mesh_name:
                                new_mesh.GetPrim().GetAttribute("visibility").Set(
                                    "invisible"
                                )
                                print(
                                    f"    Hidden newly created mesh with wall_shader: {new_mesh_name}"
                                )

                        except Exception as e:
                            print(
                                f"Error processing subset {i} ({subset_name if 'subset_name' in locals() else 'unknown'}): {e}"
                            )
                            import traceback

                            traceback.print_exc()
                            continue

                    # Optional: Clean up - hide the original mesh
                    prim_name = prim.GetName()

                    # Check if this prim should be hidden
                    should_hide = True
                    skip_reason = ""

                    if "_wall" in prim_name:
                        should_hide = False
                        skip_reason = "contains '_wall'"
                    elif "_floor" in prim_name:
                        should_hide = False
                        skip_reason = "contains '_floor'"
                    elif "_ceiling" in prim_name:
                        should_hide = False
                        skip_reason = "contains '_ceiling'"

                    if should_hide:
                        prim.GetAttribute("visibility").Set("invisible")
                        print(f"    Hidden original mesh: {prim_name}")
                    else:
                        print(f"    Kept visible ({skip_reason}): {prim_name}")

                    # Optional: Remove the original geomSubset prims
                    # for subset in geom_subsets:
                    #     subset_prim = subset.GetPrim()
                    #     if subset_prim:
                    #         omni.kit.commands.execute("DeletePrimsCommand", paths=[str(subset_prim.GetPath())])

    print(
        f"Successfully created {len(created_meshes)} separate meshes from geomSubsets"
    )
    return created_meshes


def copy_transform_attributes(source_prim, target_prim):
    """Copy transform attributes from source to target prim."""
    transform_attrs = [
        "xformOp:translate",
        "xformOp:rotateXYZ",
        "xformOp:scale",
        "xformOp:orient",
    ]

    for attr_name in transform_attrs:
        if source_prim.HasAttribute(attr_name):
            source_attr = source_prim.GetAttribute(attr_name)
            target_attr = target_prim.CreateAttribute(
                attr_name, source_attr.GetTypeName()
            )
            target_attr.Set(source_attr.Get())


def resolve_scale_issues_with_metrics_assembler() -> None:
    """Enable and execute metrics assembler to resolve scale issues in the stage."""
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.usd.metrics.assembler"):
        ext_manager.set_extension_enabled_immediate("omni.usd.metrics.assembler", True)
    from omni.metrics.assembler.core import get_metrics_assembler_interface

    stage_id = omni.usd.get_context().get_stage_id()
    get_metrics_assembler_interface().resolve_stage(stage_id)


def get_matching_prim_location(match_string, root_path=None):
    prim = find_matching_prims(
        match_strings=[match_string],
        root_path=root_path,
        prim_type="Xform",
        first_match_only=True,
    )
    if prim is None:
        print(f"[SDG-Infinigen] Could not find matching prim, returning (0, 0, 0)")
        return (0, 0, 0)
    if prim.HasAttribute("xformOp:translate"):
        return prim.GetAttribute("xformOp:translate").Get()
    elif prim.HasAttribute("xformOp:transform"):
        return prim.GetAttribute("xformOp:transform").Get().ExtractTranslation()
    else:
        print(
            f"[SDG-Infinigen] Could not find location attribute for '{prim.GetPath()}', returning (0, 0, 0)"
        )
        return (0, 0, 0)


def offset_range(
    range_coords: tuple[float, float, float, float, float, float],
    offset: tuple[float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Offset the min and max coordinates of a range by the specified offset."""
    return (
        range_coords[0] + offset[0],  # min_x
        range_coords[1] + offset[1],  # min_y
        range_coords[2] + offset[2],  # min_z
        range_coords[3] + offset[0],  # max_x
        range_coords[4] + offset[1],  # max_y
        range_coords[5] + offset[2],  # max_z
    )


def randomize_poses(
    prims: list[Usd.Prim],
    location_range: tuple[float, float, float, float, float, float],
    rotation_range: tuple[float, float],
    scale_range: tuple[float, float],
) -> None:
    """Randomize the location, rotation, and scale of a list of prims within specified ranges."""
    for prim in prims:
        rand_loc = (
            random.uniform(location_range[0], location_range[3]),
            random.uniform(location_range[1], location_range[4]),
            random.uniform(location_range[2], location_range[5]),
        )
        rand_rot = (
            random.uniform(rotation_range[0], rotation_range[1]),
            random.uniform(rotation_range[0], rotation_range[1]),
            random.uniform(rotation_range[0], rotation_range[1]),
        )
        rand_scale = random.uniform(scale_range[0], scale_range[1])
        set_transform_attributes(
            prim,
            location=rand_loc,
            rotation=rand_rot,
            scale=(rand_scale, rand_scale, rand_scale),
        )


def find_defect_spawn_planes(
    max_num_planes: int,
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Find the defect spawn planes in the stage."""
    x_spawn_planes = []
    y_spawn_planes = []

    for i in range(1, max_num_planes + 1):
        x_path = f"/Environment/dining_room_0_0_wall/x_defect_spawn_plane_0{i}"
        y_path = f"/Environment/dining_room_0_0_wall/y_defect_spawn_plane_0{i}"

        stage = omni.usd.get_context().get_stage()
        x_prim = stage.GetPrimAtPath(x_path)
        y_prim = stage.GetPrimAtPath(y_path)
        if x_prim and x_prim.IsA(UsdGeom.Mesh):
            x_spawn_planes.append(x_prim)
        if y_prim and y_prim.IsA(UsdGeom.Mesh):
            y_spawn_planes.append(y_prim)

    return x_spawn_planes, y_spawn_planes


def place_defect_planes_randomly_on_wall(target_assets):
    """Randomly place existing planes on valid wall areas (excluding holes)."""
    # wall_mesh = UsdGeom.Mesh(wall_mesh_prim)

    wall_mesh = find_matching_prims(
        match_strings=["dining_room_0_0_wall"],
        root_path="/Environment",
        prim_type="Mesh",
        first_match_only=True,
    )

    wall_xform = find_matching_prims(
        match_strings=["dining_room_0_0_wall"],
        root_path="/Environment",
        prim_type="Xform",
        first_match_only=True,
    )

    print(target_assets, flush=True)

    # Get wall mesh data
    points = wall_mesh.GetAttribute("points").Get()
    face_vertex_counts = wall_mesh.GetAttribute("faceVertexCounts").Get()
    face_vertex_indices = wall_mesh.GetAttribute("faceVertexIndices").Get()
    normals = wall_mesh.GetAttribute("normals").Get()
    hole_indices = wall_mesh.GetAttribute("holeIndices").Get()

    # Get existing planes
    existing_planes = target_assets

    # Get valid face centers (excluding holes)
    valid_face_centers = []
    valid_face_normals = []
    face_start = 0

    for face_idx in range(len(face_vertex_counts)):
        # Skip holes if specified
        if hole_indices and face_idx in hole_indices:
            face_start += face_vertex_counts[face_idx]
            print("skipping hole", flush=True)
            continue

        face_vertex_count = face_vertex_counts[face_idx]
        face_vertices = face_vertex_indices[face_start : face_start + face_vertex_count]

        # Calculate face center
        face_center = Gf.Vec3f(0, 0, 0)
        for vertex_idx in face_vertices:
            face_center += points[vertex_idx]
        face_center /= len(face_vertices)

        # Calculate face area to filter out small edge faces
        if face_vertex_count >= 3:
            v0 = points[face_vertices[0]]
            v1 = points[face_vertices[1]]
            v2 = points[face_vertices[2]]

            edge1 = v1 - v0
            edge2 = v2 - v0
            face_area = edge1.cross(edge2).GetLength() * 0.5

            # Skip faces that are too small (likely edges)
            if face_area < 0.01:  # Adjust this threshold as needed
                face_start += face_vertex_count
                print(
                    f"skipping small edge face {face_idx} with area {face_area}",
                    flush=True,
                )
                continue

        # Get face normal
        face_normal = normals[face_idx]

        valid_face_centers.append(face_center)
        valid_face_normals.append(face_normal)
        face_start += face_vertex_count

    # Shuffle valid faces for random placement
    face_indices = list(range(len(valid_face_centers)))
    random.shuffle(face_indices)

    # Determine how many planes to place
    num_planes_to_place = len(existing_planes)

    # Get wall's world transform
    wall_transform = wall_xform.GetAttribute("xformOp:translate").Get()
    wall_rotation = wall_xform.GetAttribute("xformOp:orient").Get()
    wall_scale = wall_xform.GetAttribute("xformOp:scale").Get()

    print(
        f"Wall transform: {wall_transform}, rotation: {wall_rotation}, scale: {wall_scale}",
        flush=True,
    )

    # Place planes randomly on valid faces
    for i in range(num_planes_to_place):
        plane_prim = existing_planes[i]
        face_idx = face_indices[i]

        face_center = valid_face_centers[face_idx]
        face_normal = valid_face_normals[face_idx]

        # Transform face center to world coordinates
        # Apply wall's rotation and translation to the face center
        if wall_rotation:
            # Rotate the face center by wall's rotation
            rotated_center = wall_rotation.Transform(face_center)
        else:
            rotated_center = face_center

        # Add wall's translation
        world_center = Gf.Vec3f(
            rotated_center[0] + wall_transform[0],
            rotated_center[1] + wall_transform[1],
            rotated_center[2] + wall_transform[2],
        )

        plane_prim.CreateAttribute(
            "xformOp:translate", Sdf.ValueTypeNames.Vector3f
        ).Set(world_center)

        # Orient plane to match surface normal
        up_vector = Gf.Vec3d(0, 0, 1)
        face_normal_d = Gf.Vec3d(face_normal[0], face_normal[1], face_normal[2])
        if abs(face_normal_d.GetDot(up_vector)) > 0.9:
            up_vector = Gf.Vec3d(0, 1, 0)

        rotation = Gf.Rotation(up_vector, face_normal_d)
        quat_d = rotation.GetQuat()
        imag = quat_d.GetImaginary()
        quat_f = Gf.Quatf(quat_d.GetReal(), Gf.Vec3f(imag[0], imag[1], imag[2]))
        plane_prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatf).Set(
            quat_f
        )

        print(f"Randomly placed plane {i} at face {face_idx}")


def register_defect_scatter_randomizer(
    target_assets: list[Usd.Prim],
    collision_check: bool = False,
    scale_range: tuple[float, float] = (0.95, 1.15),
) -> None:
    """Register a replicator graph randomizer."""
    with rep.trigger.on_custom_event(event_name="randomize_defect_scatter"):
        random.shuffle(target_assets)
        half_size = len(target_assets) // 2
        first_half = target_assets[:half_size]
        second_half = target_assets[half_size:]

        fist_target_prims_paths = [prim.GetPath() for prim in first_half]
        first_target_prims_group = rep.create.group(fist_target_prims_paths)
        second_target_prims_paths = [prim.GetPath() for prim in second_half]
        second_target_prims_group = rep.create.group(second_target_prims_paths)

        # Get spawn planes for defect scatter randomizer TODO: FIXXXXX
        x_spawn_planes, y_spawn_planes = find_defect_spawn_planes(max_num_planes=20)
        print(f"[SDG-Infinigen] Found {len(x_spawn_planes)} x spawn planes")
        print(f"[SDG-Infinigen] Found {len(y_spawn_planes)} y spawn planes")

        with first_target_prims_group:
            rep.randomizer.scatter_2d(
                surface_prims=x_spawn_planes,
                check_for_collisions=collision_check,
            )
            rep.modify.pose(
                rotation=rep.distribution.uniform((90, 0, 0), (90, 0, 0)),
                scale=rep.distribution.uniform(scale_range[0], scale_range[1]),
            )
        with second_target_prims_group:
            rep.randomizer.scatter_2d(
                surface_prims=y_spawn_planes,
                check_for_collisions=collision_check,
            )
            rep.modify.pose(
                rotation=rep.distribution.uniform((0, 90, 0), (0, 90, 0)),
                scale=rep.distribution.uniform(scale_range[0], scale_range[1]),
            )


def run_simulation(num_frames: int, render: bool = True) -> None:
    """Run a simulation for a specified number of frames, optionally without rendering."""
    if render:
        # Start the timeline and advance the app, this will render the physics simulation results every frame
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_start_time(0)
        timeline.set_end_time(1000000)
        timeline.set_looping(False)
        timeline.play()
        for _ in range(num_frames):
            omni.kit.app.get_app().update()
        timeline.pause()
    else:
        # Run the physics simulation steps without advancing the app
        stage = omni.usd.get_context().get_stage()
        physx_scene = None

        # Search for or create a physics scene
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
                break

        if physx_scene is None:
            physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(
                stage.GetPrimAtPath("/PhysicsScene")
            )

        # Get simulation parameters
        physx_dt = 1 / physx_scene.GetTimeStepsPerSecondAttr().Get()
        physx_sim_interface = omni.physx.get_physx_simulation_interface()

        # Run physics simulation for each frame
        for _ in range(num_frames):
            physx_sim_interface.simulate(physx_dt, 0)
            physx_sim_interface.fetch_results()


def register_dome_light_randomizer() -> None:
    """Register a replicator graph randomizer for dome lights using various sky textures."""
    assets_root_path = get_assets_root_path()
    dome_textures = [
        assets_root_path + "/NVIDIA/Assets/Skies/Cloudy/champagne_castle_1_4k.hdr",
        assets_root_path
        + "/NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Clear/evening_road_01_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Clear/mealie_road_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Clear/qwantani_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Evening/evening_road_01_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Night/kloppenheim_02_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Night/moonlit_golf_4k.hdr",
    ]
    with rep.trigger.on_custom_event(event_name="randomize_dome_lights"):
        rep.create.light(
            light_type="Dome", texture=rep.distribution.choice(dome_textures)
        )


def register_shape_distractors_color_randomizer(
    shape_distractors: list[Usd.Prim],
) -> None:
    """Register a replicator graph randomizer to change colors of shape distractors."""
    with rep.trigger.on_custom_event(event_name="randomize_shape_distractor_colors"):
        shape_distractors_paths = [prim.GetPath() for prim in shape_distractors]
        shape_distractors_group = rep.create.group(shape_distractors_paths)
        with shape_distractors_group:
            rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))


def randomize_lights(
    location_range: tuple[float, float, float, float, float, float] | None = None,
    color_range: tuple[float, float, float, float, float, float] | None = None,
    intensity_range: tuple[float, float] | None = None,
) -> None:
    """Randomize location, color, and intensity of specified lights within given ranges."""
    stage = omni.usd.get_context().get_stage()
    point_lamp_lights = []
    for prim in stage.Traverse():
        if prim.GetName().startswith("PointLampFactory"):
            point_lamp_lights.append(prim)

    print(f"[SDG-Infinigen] Found {len(point_lamp_lights)} lights")

    for light in point_lamp_lights:
        # Randomize the location of the light
        if location_range is not None:
            rand_loc = (
                random.uniform(location_range[0], location_range[3]),
                random.uniform(location_range[1], location_range[4]),
                random.uniform(location_range[2], location_range[5]),
            )
            set_transform_attributes(light, location=rand_loc)

        # Randomize the color of the light
        if color_range is not None:
            rand_color = (
                random.uniform(color_range[0], color_range[3]),
                random.uniform(color_range[1], color_range[4]),
                random.uniform(color_range[2], color_range[5]),
            )
            light.GetAttribute("inputs:color").Set(rand_color)

        # Randomize the intensity of the light
        if intensity_range is not None:
            rand_intensity = random.uniform(intensity_range[0], intensity_range[1])
            light.GetAttribute("inputs:intensity").Set(rand_intensity)


def setup_writer(config: dict) -> None:
    """Setup a writer based on configuration settings, initializing with specified arguments."""
    writer_type = config.get("type", None)
    if writer_type is None:
        print("[Infinigen-SDG] No writer type specified. No writer will be used.")
        return None

    try:
        writer = rep.writers.get(writer_type)
    except Exception as e:
        print(
            f"[Infinigen-SDG] Writer type '{writer_type}' not found. No writer will be used. Error: {e}"
        )
        return None

    writer_kwargs = config.get("kwargs", {})
    if out_dir := writer_kwargs.get("output_dir"):
        # If not an absolute path, make path relative to the current working directory
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(os.getcwd(), out_dir)
            writer_kwargs["output_dir"] = out_dir

    writer.initialize(**writer_kwargs)
    return writer


def add_distant_lights_for_windows():
    """Add distant lights pointing at each WindowFactory prim to simulate exterior lighting."""
    stage = omni.usd.get_context().get_stage()

    # Find all WindowFactory prims
    window_factories = []
    for prim in stage.Traverse():
        if prim.GetName().startswith("WindowFactory"):
            if prim.IsA(UsdGeom.Xform):
                window_factories.append(prim)
    print(
        f"Found {len(window_factories)} WindowFactory prims, adding distant lights..."
    )

    for i, window_prim in enumerate(window_factories):
        try:
            # Get the window's position
            if window_prim.HasAttribute("xformOp:translate"):
                window_pos = window_prim.GetAttribute("xformOp:translate").Get()
            else:
                window_pos = (0, 0, 0)

            # Create a distant light positioned outside and pointing at the window
            light_name = f"DistantLight_Window_{i:02d}"
            light_path = f"/World/{light_name}"

            # Position the light outside the window (adjust these values based on your scene)
            light_pos = (
                window_pos[0] + random.uniform(-20, 20),  # Random X offset
                window_pos[1] + random.uniform(-20, 20),  # Random Y offset
                window_pos[2] + random.uniform(10, 30),  # Above the window
            )

            # Create the distant light
            distant_light = prim_utils.create_prim(light_path, "DistantLight")

            # Set the light's position
            distant_light.CreateAttribute(
                "xformOp:translate", Sdf.ValueTypeNames.Vector3f
            ).Set(light_pos)

            # Point the light at the window
            direction_to_window = (
                window_pos[0] - light_pos[0],
                window_pos[1] - light_pos[1],
                window_pos[2] - light_pos[2],
            )
            # Normalize the direction
            length = math.sqrt(sum(d * d for d in direction_to_window))
            normalized_direction = tuple(d / length for d in direction_to_window)

            # Set the light's direction (distant lights use direction, not position)
            distant_light.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(
                0.5
            )  # Light cone angle
            distant_light.CreateAttribute(
                "inputs:intensity", Sdf.ValueTypeNames.Float
            ).Set(random.uniform(5000, 15000))
            distant_light.CreateAttribute(
                "inputs:color", Sdf.ValueTypeNames.Color3f
            ).Set(
                (1.0, 0.95, 0.8)
            )  # Warm sunlight color
            distant_light.CreateAttribute(
                "inputs:exposure", Sdf.ValueTypeNames.Float
            ).Set(0.0)

            print(
                f"Created distant light {light_name} at {light_pos} pointing at window {i}"
            )

        except Exception as e:
            print(f"Error creating distant light for window {i}: {e}")

    print(f"Successfully created {len(window_factories)} distant lights for windows")


def add_sun_light_for_windows():
    """Add a single large distant light (sun) that illuminates all windows."""
    stage = omni.usd.get_context().get_stage()

    # Find all WindowFactory prims to calculate the center
    window_factories = []
    for prim in stage.Traverse():
        if prim.GetName().startswith("WindowFactory"):
            if prim.IsA(UsdGeom.Xform):
                window_factories.append(prim)

    if not window_factories:
        print("No WindowFactory prims found")
        return

    # Calculate the center of all windows
    center_x = sum(
        prim.GetAttribute("xformOp:translate").Get()[0]
        for prim in window_factories
        if prim.HasAttribute("xformOp:translate")
    ) / len(window_factories)
    center_y = sum(
        prim.GetAttribute("xformOp:translate").Get()[1]
        for prim in window_factories
        if prim.HasAttribute("xformOp:translate")
    ) / len(window_factories)
    center_z = sum(
        prim.GetAttribute("xformOp:translate").Get()[2]
        for prim in window_factories
        if prim.HasAttribute("xformOp:translate")
    ) / len(window_factories)

    window_center = (center_x, center_y, center_z)

    # Create a large distant light (sun)
    sun_light = prim_utils.create_prim(
        "/World/SunLight",
        "DistantLight",
    )

    # Position the sun high above and to the side
    sun_pos = (
        window_center[0] + random.uniform(-50, 50),
        window_center[1] + random.uniform(-50, 50),
        window_center[2] + random.uniform(80, 120),
    )

    # Set the sun's position
    sun_light.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Vector3f).Set(
        sun_pos
    )

    # Point the sun down at the windows
    sun_direction = (
        window_center[0] - sun_pos[0],
        window_center[1] - sun_pos[1],
        window_center[2] - sun_pos[2],
    )

    # Set sun properties
    sun_light.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(0.5)
    sun_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(20000)
    sun_light.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(
        (1.0, 0.95, 0.8)
    )  # Warm sunlight
    sun_light.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(0.0)

    print(f"Created sun light at {sun_pos} pointing at window center {window_center}")


def add_sun_lights_for_scene():
    """Add 4 distant lights around the Environment with proper quaternion orientations."""
    stage = omni.usd.get_context().get_stage()

    # Find the /Environment path and get its transform
    environment_prim = stage.GetPrimAtPath("/Environment")
    if not environment_prim:
        print("ERROR: Could not find /Environment path")
        return

    # Get the environment's transform
    env_transform = (0, 0, 0)  # Default if no transform
    env_rotation = (0, 0, 0)  # Default if no rotation

    if environment_prim.HasAttribute("xformOp:translate"):
        env_transform = environment_prim.GetAttribute("xformOp:translate").Get()
        print(f"Found Environment transform: {env_transform}")
    else:
        print("No transform found on /Environment, using (0,0,0)")

    if environment_prim.HasAttribute("xformOp:rotateXYZ"):
        env_rotation = environment_prim.GetAttribute("xformOp:rotateXYZ").Get()
        print(f"Found Environment rotation: {env_rotation}")
    else:
        print("No rotation found on /Environment, using (0,0,0)")

    # Create 4 distant lights at specified positions relative to environment
    light_configs = [
        {
            "name": "DistantLight_NW",
            "position": (-30, 0, 30),
            "orientation": (
                0.56099,
                -0.43046,
                -0.43046,
                0.56099,
            ),  # Changed from rotation to orientation
            "description": "Northwest light",
        },
        {
            "name": "DistantLight_N",
            "position": (0, 30, 30),
            "orientation": (
                0.79335,
                -0.60876,
                0,
                0,
            ),  # Changed from rotation to orientation
            "description": "North light",
        },
        {
            "name": "DistantLight_NE",
            "position": (30, 0, 30),
            "orientation": (
                0.56099,
                0.43046,
                0.43046,
                0.56099,
            ),  # Changed from rotation to orientation
            "description": "Northeast light",
        },
        {
            "name": "DistantLight_S",
            "position": (0, -30, 30),
            "orientation": (
                0.79335,
                0.60876,
                0,
                0,
            ),  # Changed from rotation to orientation
            "description": "South light",
        },
    ]

    created_lights = []

    for light_config in light_configs:
        try:
            # Calculate absolute position by adding environment transform
            abs_position = (
                env_transform[0] + light_config["position"][0],
                env_transform[1] + light_config["position"][1],
                env_transform[2] + light_config["position"][2],
            )

            # Create the distant light
            light_path = f"/World/{light_config['name']}"
            distant_light = prim_utils.create_prim(light_path, "DistantLight")

            # Set position
            distant_light.CreateAttribute(
                "xformOp:translate", Sdf.ValueTypeNames.Vector3f
            ).Set(abs_position)

            # Get quaternion values from config (w, x, y, z format)
            quat_values = light_config["orientation"]

            # Create proper Gf.Quatd object (w, x, y, z format)
            quat = Gf.Quatd(
                quat_values[0], Gf.Vec3d(quat_values[1], quat_values[2], quat_values[3])
            )

            # Set orientation using quaternion with double precision
            distant_light.CreateAttribute(
                "xformOp:orient", Sdf.ValueTypeNames.Quatd
            ).Set(quat)

            # Set light properties
            distant_light.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(
                0.5
            )
            distant_light.CreateAttribute(
                "inputs:intensity", Sdf.ValueTypeNames.Float
            ).Set(random.uniform(8000, 15000))
            distant_light.CreateAttribute(
                "inputs:color", Sdf.ValueTypeNames.Color3f
            ).Set(
                (1.0, 0.95, 0.8)
            )  # Warm sunlight
            distant_light.CreateAttribute(
                "inputs:exposure", Sdf.ValueTypeNames.Float
            ).Set(0.0)

            created_lights.append(distant_light)
            print(
                f"Created {light_config['name']} at {abs_position} with quaternion {quat} ({light_config['description']})"
            )

        except Exception as e:
            print(f"ERROR: Failed to create {light_config['name']}: {e}")
            import traceback

            traceback.print_exc()

    print(
        f"Successfully created {len(created_lights)} distant lights around /Environment"
    )
    return created_lights


def load_apartment_materials_to_stage():
    """Load defect MDL materials into the stage."""
    import os
    import omni.kit.commands

    wood_material_paths = []
    paint_plaster_material_paths = []
    glass_material_paths = []

    # Get the path to the defects folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    glass_folder = os.path.join(script_dir, "usds", "mdl", "glass")
    paint_plaster_folder = os.path.join(script_dir, "usds", "mdl", "paint_plaster")
    wood_folder = os.path.join(script_dir, "usds", "mdl", "wood")

    # Create the Looks prim if it doesn't exist
    stage = omni.usd.get_context().get_stage()
    looks_prim = stage.GetPrimAtPath("/World/Looks")
    if not looks_prim:
        looks_prim = stage.DefinePrim("/World/Looks", "Xform")

    # Find all MDL files in the defects folder
    import glob

    mdl_pattern_glass = os.path.join(glass_folder, "*.mdl")
    mdl_pattern_paint_plaster = os.path.join(paint_plaster_folder, "*.mdl")
    mdl_pattern_wood = os.path.join(wood_folder, "*.mdl")
    mdl_files_glass = glob.glob(mdl_pattern_glass)
    mdl_files_paint_plaster = glob.glob(mdl_pattern_paint_plaster)
    mdl_files_wood = glob.glob(mdl_pattern_wood)

    if not mdl_files_glass or not mdl_files_paint_plaster or not mdl_files_wood:
        print(
            f"[SDG-Infinigen] No MDL files found in {glass_folder}, {paint_plaster_folder}, {wood_folder}"
        )
        return

    # Load each defect MDL file
    for glass_path in mdl_files_glass:
        glass_filename = os.path.basename(glass_path)
        glass_name = os.path.splitext(glass_filename)[0]  # Remove .mdl extension
        print(f"[SDG-Infinigen] Loading glass material: {glass_filename}")
        material_prim_path = f"/World/Looks/{glass_name}"
        glass_material_paths.append(material_prim_path)

        # Load the MDL material
        try:
            omni.kit.commands.execute(
                "CreateMdlMaterialPrim",
                mtl_url=f"file://{glass_path}",
                mtl_name=glass_name,
                mtl_path=material_prim_path,
            )
            print(f"[SDG-Infinigen] Loaded glass material: {glass_filename}")
        except Exception as e:
            print(
                f"[SDG-Infinigen] Failed to load glass material {glass_filename}: {e}"
            )

    for paint_plaster_path in mdl_files_paint_plaster:

        paint_plaster_filename = os.path.basename(paint_plaster_path)
        paint_plaster_name = os.path.splitext(paint_plaster_filename)[
            0
        ]  # Remove .mdl extension
        print(
            f"[SDG-Infinigen] Loading paint plaster material: {paint_plaster_filename}"
        )
        material_prim_path = f"/World/Looks/{paint_plaster_name}"
        paint_plaster_material_paths.append(material_prim_path)
        try:

            omni.kit.commands.execute(
                "CreateMdlMaterialPrim",
                mtl_url=f"file://{paint_plaster_path}",
                mtl_name=paint_plaster_name,
                mtl_path=material_prim_path,
            )

            print(
                f"[SDG-Infinigen] Loaded paint plaster material: {paint_plaster_filename}"
            )
        except Exception as e:
            print(
                f"[SDG-Infinigen] Failed to load paint plaster material {paint_plaster_filename}: {e}"
            )

    for wood_path in mdl_files_wood:

        wood_filename = os.path.basename(wood_path)
        wood_name = os.path.splitext(wood_filename)[0]  # Remove .mdl extension
        print(f"[SDG-Infinigen] Loading wood material: {wood_filename}")
        material_prim_path = f"/World/Looks/{wood_name}"
        wood_material_paths.append(material_prim_path)
        try:
            omni.kit.commands.execute(
                "CreateMdlMaterialPrim",
                mtl_url=f"file://{wood_path}",
                mtl_name=wood_name,
                mtl_path=material_prim_path,
            )

            print(f"[SDG-Infinigen] Loaded wood material: {wood_filename}")
        except Exception as e:
            print(f"[SDG-Infinigen] Failed to load wood material {wood_filename}: {e}")

    return wood_material_paths, paint_plaster_material_paths, glass_material_paths


def register_randomize_custom_textures_to_scene(
    # wood_material_paths, paint_plaster_material_paths, glass_material_paths
):
    glass_material_paths = [
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Glass/Glass_Clear.mdl",
    ]
    wood_material_paths = [
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Wood/Laminate_Oak.mdl",
    ]
    floor_material_paths = [
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Wood/Wood_Tiles_Walnut.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Wood/Wood_Tiles_Oak_Mountain.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Wood/Wood_Tiles_Ash.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Wood/Wood_Tiles_Fineline.mdl",
    ]
    paint_plaster_material_paths = [
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Paint/Chalk_Paint.mdl",
    ]
    plastic_material_paths = [
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Plastic/Polyethylene_Opaque.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Plastic/Polypropylene_Opaque.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Plastic/Polycarbonate_Opaque.mdl",
    ]
    ceiling_light_material_paths = [
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Metal/Aluminum.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Metal/Stainless_Steel.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Paint/Hammer_Paint.mdl",
    ]
    curtain_material_paths = [
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Fabric/Cashmere_Wool_Suiting.mdl",
    ]

    """Add custom textures to the scene."""
    with rep.trigger.on_custom_event(event_name="randomize_custom_textures_to_scene"):
        stage = omni.usd.get_context().get_stage()
        window_glass_meshes = []
        curtain_meshes = []
        for prim in stage.Traverse():
            if (
                prim.IsA(UsdGeom.Mesh)
                and "shader_window_glass" in prim.GetName().lower()
            ):
                window_glass_meshes.append(prim)
            elif prim.IsA(UsdGeom.Mesh) and (
                "WindowFactory" in prim.GetName()
                and "shader_rough_plastic" in prim.GetName()
            ):
                curtain_meshes.append(prim)

        rep.randomizer.materials(
            materials=glass_material_paths, input_prims=window_glass_meshes
        )
        rep.randomizer.materials(
            materials=curtain_material_paths, input_prims=curtain_meshes
        )
        wood_meshes = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and (
                "shader_wood" in prim.GetName().lower()
                or "shader_shelves_wood" in prim.GetName().lower()
            ):
                wood_meshes.append(prim)

        metal_meshes = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and (
                "shader" in prim.GetName().lower() and "metal" in prim.GetName().lower()
            ):
                metal_meshes.append(prim)
        rep.randomizer.materials(
            materials=ceiling_light_material_paths, input_prims=metal_meshes
        )
        rep.randomizer.materials(materials=wood_material_paths, input_prims=wood_meshes)

        paint_plaster_meshes = []
        ceiling_meshes = []
        floor_meshes = []
        skirt_board_meshes = []
        ceiling_light_meshes = []
        door_meshes = []
        door_stuff_meshes = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and "_wall" in prim.GetName():
                paint_plaster_meshes.append(prim)
            elif prim.IsA(UsdGeom.Mesh) and prim.GetName().endswith("_floor"):
                floor_meshes.append(prim)
            elif prim.IsA(UsdGeom.Mesh) and (
                "skirtingboard" not in prim.GetName().lower()
                and prim.GetName().endswith("_ceiling")
            ):
                ceiling_meshes.append(prim)
            elif prim.IsA(UsdGeom.Mesh) and "skirtingboard" in prim.GetName().lower():
                skirt_board_meshes.append(prim)
            elif prim.IsA(UsdGeom.Mesh) and prim.GetName().startswith(
                "CeilingLightFactory_"
            ):
                ceiling_light_meshes.append(prim)
            elif prim.IsA(UsdGeom.Mesh) and (
                "DoorFactory" in prim.GetName() and not prim.GetName().endswith("001")
            ):
                door_meshes.append(prim)
            elif prim.IsA(UsdGeom.Mesh) and (
                "DoorFactory" in prim.GetName() and prim.GetName().endswith("001")
            ):
                door_stuff_meshes.append(prim)
        rep.randomizer.materials(
            materials=paint_plaster_material_paths, input_prims=paint_plaster_meshes
        )

        rep.randomizer.materials(
            materials=paint_plaster_material_paths, input_prims=ceiling_meshes
        )

        # Set material binding strength to 'strongerThanDescendants' for wall meshes
        for wall_prim in paint_plaster_meshes:
            try:
                if wall_prim.HasAttribute("material:binding:strength"):
                    wall_prim.GetAttribute("material:binding:strength").Set(
                        UsdShade.Tokens.strongerThanDescendants
                    )
                    print(
                        f"    Updated material binding strength for {wall_prim.GetName()}"
                    )
                else:
                    # Create the attribute if it doesn't exist
                    strength_attr = wall_prim.CreateAttribute(
                        "material:binding:strength", Sdf.ValueTypeNames.Token
                    )
                    strength_attr.Set(UsdShade.Tokens.strongerThanDescendants)
                    print(
                        f"    Created and set material binding strength for {wall_prim.GetName()}"
                    )
            except Exception as e:
                print(
                    f"    ERROR: Failed to set material binding strength for {wall_prim.GetName()}: {e}"
                )

        rep.randomizer.materials(
            materials=floor_material_paths, input_prims=floor_meshes
        )
        rep.randomizer.materials(
            materials=plastic_material_paths, input_prims=skirt_board_meshes
        )
        rep.randomizer.materials(
            materials=ceiling_light_material_paths, input_prims=ceiling_light_meshes
        )
        rep.randomizer.materials(materials=wood_material_paths, input_prims=door_meshes)
        rep.randomizer.materials(
            materials=ceiling_light_material_paths, input_prims=door_stuff_meshes
        )


# /Replicator/Looks/Chalk_Paint/Shader


def set_window_glass_shadow_attributes():
    """Find meshes with 'shader_window_glass' in their name and set doNotCastShadows to false."""
    stage = omni.usd.get_context().get_stage()

    # Find all meshes with 'shader_window_glass' in their name
    window_glass_meshes = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and "shader_window_glass" in prim.GetName().lower():
            window_glass_meshes.append(prim)

    print(
        f"Found {len(window_glass_meshes)} meshes with 'shader_window_glass' in their name"
    )

    for i, mesh_prim in enumerate(window_glass_meshes):
        try:
            # Create the primvars:doNotCastShadows attribute
            # Note: primvars are typically created on the prim, not the mesh schema
            prim = mesh_prim.GetPrim()

            # Check if the attribute already exists
            if prim.HasAttribute("primvars:doNotCastShadows"):
                print(
                    f"    Updating existing doNotCastShadows attribute for {mesh_prim.GetName()}"
                )
                prim.GetAttribute("primvars:doNotCastShadows").Set(True)
            else:
                print(
                    f"    Creating new doNotCastShadows attribute for {mesh_prim.GetName()}"
                )
                # Create the attribute as a boolean
                do_not_cast_shadows_attr = prim.CreateAttribute(
                    "primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool
                )
                # Set it to False (so shadows WILL be cast)
                do_not_cast_shadows_attr.Set(True)

            print(
                f"    Successfully set doNotCastShadows=True for {mesh_prim.GetName()}"
            )

        except Exception as e:
            print(
                f"    ERROR: Failed to set doNotCastShadows for {mesh_prim.GetName()}: {e}"
            )

    print(f"Successfully processed {len(window_glass_meshes)} window glass meshes")
    return window_glass_meshes


def set_floor_and_wall_shadow_attributes():
    """Find meshes with 'shader_window_glass' in their name and set doNotCastShadows to false."""
    stage = omni.usd.get_context().get_stage()

    # Find all meshes with 'shader_window_glass' in their name
    floor_wall_ceiling_meshes = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and (
            "_wall" in prim.GetName().lower()
            or "_floor" in prim.GetName().lower()
            or "_ceiling" in prim.GetName().lower()
        ):
            floor_wall_ceiling_meshes.append(prim)

    for i, mesh_prim in enumerate(floor_wall_ceiling_meshes):
        try:
            # Create the primvars:doNotCastShadows attribute
            # Note: primvars are typically created on the prim, not the mesh schema
            prim = mesh_prim.GetPrim()

            # Check if the attribute already exists
            if prim.HasAttribute("primvars:invisibleToSecondayRays"):
                print(
                    f"    Updating existing invisibleToSecondayRays attribute for {mesh_prim.GetName()}"
                )
                prim.GetAttribute("primvars:invisibleToSecondayRays").Set(True)
            else:
                print(
                    f"    Creating new invisibleToSecondayRays attribute for {mesh_prim.GetName()}"
                )
                # Create the attribute as a boolean
                do_not_cast_shadows_attr = prim.CreateAttribute(
                    "primvars:invisibleToSecondayRays", Sdf.ValueTypeNames.Bool
                )
                # Set it to False (so shadows WILL be cast)
                do_not_cast_shadows_attr.Set(True)

            print(
                f"    Successfully set invisibleToSecondayRays=True for {mesh_prim.GetName()}"
            )

        except Exception as e:
            print(
                f"    ERROR: Failed to set invisibleToSecondayRays for {mesh_prim.GetName()}: {e}"
            )

    print(
        f"Successfully processed {len(floor_wall_ceiling_meshes)} window glass meshes"
    )
    return floor_wall_ceiling_meshes


def turn_off_env_light():
    """Turn off all environment lights."""
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.GetName().startswith("env_light"):
            prim.GetAttribute("visibility").Set("invisible")

    print(f"Successfully turned off the environment light")
    return


def set_point_lamp_intensities(give_intensity: int, give_color_temperature: int):
    """Find PointLampFactory lights and set their intensity to 20000."""
    stage = omni.usd.get_context().get_stage()

    # Find all PointLampFactory lights
    point_lamp_lights = []
    for prim in stage.Traverse():
        if prim.GetName().startswith("PointLampFactory"):
            point_lamp_lights.append(prim)

    print(f"Found {len(point_lamp_lights)} PointLampFactory lights")

    updated_lights = []

    for i, light_prim in enumerate(point_lamp_lights):
        try:
            light_name = light_prim.GetName()
            light_path = str(light_prim.GetPath())

            print(f"    Processing light {i+1}: {light_name} at {light_path}")

            # Check if the light has an intensity attribute
            if light_prim.HasAttribute("inputs:intensity"):
                # Get current intensity
                current_intensity = light_prim.GetAttribute("inputs:intensity").Get()
                print(f"        Current intensity: {current_intensity}")

                # Set new intensity
                light_prim.GetAttribute("inputs:intensity").Set(give_intensity)

                # Verify the change
                new_intensity = light_prim.GetAttribute("inputs:intensity").Get()
                print(f"        Updated intensity to: {new_intensity}")

                updated_lights.append(light_prim)

            else:
                print(f"        WARNING: No intensity attribute found on {light_name}")

            try:
                if light_prim.HasAttribute("inputs:enableColorTemperature"):
                    light_prim.GetAttribute("inputs:enableColorTemperature").Set(True)
                else:
                    # Create the attribute as a boolean
                    enableColorTemperature_attr = light_prim.CreateAttribute(
                        "inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool
                    )
                    # Set it to False (so shadows WILL be cast)
                    enableColorTemperature_attr.Set(True)
            except Exception as e:
                print(
                    f"    ERROR: Failed to set inputs:enableColorTemperature for {light_prim.GetName()}: {e}"
                )

            # Check if the light has an intensity attribute
            if light_prim.HasAttribute("inputs:colorTemperature"):
                # Get current intensity
                current_color_temperature = light_prim.GetAttribute(
                    "inputs:colorTemperature"
                ).Get()
                print(f"        Current color temperature: {current_color_temperature}")

                # Set new intensity
                light_prim.GetAttribute("inputs:colorTemperature").Set(
                    give_color_temperature
                )

                # Verify the change
                new_color_temperature = light_prim.GetAttribute(
                    "inputs:colorTemperature"
                ).Get()
                print(f"        Updated color temperature to: {new_color_temperature}")

                updated_lights.append(light_prim)

            else:
                print(f"        WARNING: No intensity attribute found on {light_name}")

        except Exception as e:
            print(f"    ERROR: Failed to update {light_prim.GetName()}: {e}")
            import traceback

            traceback.print_exc()

    print(
        f"Successfully updated {len(updated_lights)} PointLampFactory lights to intensity 20000"
    )
    return updated_lights


def set_specific_camera_horizontal_aperture():
    """Set horizontalAperture to 1.0 for specific camera paths."""
    stage = omni.usd.get_context().get_stage()
    camera_paths = []
    # Define the specific camera paths you want to update
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            camera_paths.append(str(prim.GetPath()))

    updated_cameras = []

    for camera_path in camera_paths:
        try:
            camera_prim = stage.GetPrimAtPath(camera_path)

            if not camera_prim:
                print(f"    WARNING: Camera not found at path: {camera_path}")
                continue

            if not camera_prim.IsA(UsdGeom.Camera):
                print(f"    WARNING: Prim at {camera_path} is not a camera")
                continue

            print(f"    Processing camera at: {camera_path}")

            # Set horizontalAperture
            if camera_prim.HasAttribute("horizontalAperture"):
                current_aperture = camera_prim.GetAttribute("horizontalAperture").Get()
                print(f"        Current horizontalAperture: {current_aperture}")

                camera_prim.GetAttribute("horizontalAperture").Set(1.0)

                new_aperture = camera_prim.GetAttribute("horizontalAperture").Get()
                print(f"        Updated horizontalAperture to: {new_aperture}")

                updated_cameras.append(camera_prim)
            else:
                print(f"        WARNING: No horizontalAperture attribute found")

                # Create the attribute
                try:
                    aperture_attr = camera_prim.CreateAttribute(
                        "horizontalAperture", Sdf.ValueTypeNames.Float
                    )
                    aperture_attr.Set(1.0)
                    print(
                        f"        Created and set horizontalAperture attribute to 1.0"
                    )
                    updated_cameras.append(camera_prim)
                except Exception as e:
                    print(
                        f"        ERROR: Failed to create horizontalAperture attribute: {e}"
                    )

        except Exception as e:
            print(f"    ERROR: Failed to update camera at {camera_path}: {e}")

    print(
        f"Successfully updated {len(updated_cameras)} cameras to horizontalAperture 1.0"
    )
    return updated_cameras


def paint_randomizer():
    chalk_paints = [
        "Chalk_Paint_Bark",
        "Chalk_Paint_Elation",
        "Chalk_Paint_Night_Watch",
        "Chalk_Paint_Roses",
        "Chalk_Paint_Wine",
        "Chalk_Paint_Contemplation",
        "Chalk_Paint_Charcoal_Blue",
        "Chalk_Paint_Deep_Emerald",
    ]

    stage = omni.usd.get_context().get_stage()
    chalk1 = "/Replicator/Looks/Chalk_Paint"
    chalk2 = "/Replicator/Looks/Chalk_Paint_01"
    """Randomize the texture of the given prims."""
    try:
        # Note: CHALK MATERIAL
        prim = stage.GetPrimAtPath(chalk1 + "/Shader")
        # Check if the attribute already exists
        if prim.HasAttribute("info:mdl:sourceAsset:subIdentifier"):
            random_chalk_paint = random.choice(chalk_paints)
            prim.GetAttribute("info:mdl:sourceAsset:subIdentifier").Set(
                random_chalk_paint
            )
        else:
            print(f"    Creating new inputs:project_uvw attribute for {prim.GetName()}")
            # Create the attribute as a boolean
            subIdentifier_attr = prim.CreateAttribute(
                "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token
            )
            # Set it to False (so shadows WILL be cast)
            subIdentifier_attr.Set(random_chalk_paint)

        print(
            f"    Successfully set info:mdl:sourceAsset:subIdentifier for {prim.GetName()}"
        )

    except Exception as e:
        print(
            f"    ERROR: Failed to set info:mdl:sourceAsset:subIdentifier for {prim.GetName()}: {e}"
        )
    try:
        # Note: CHALK MATERIAL
        prim = stage.GetPrimAtPath(chalk2 + "/Shader")
        # Check if the attribute already exists
        if prim.HasAttribute("info:mdl:sourceAsset:subIdentifier"):
            random_chalk_paint = random.choice(chalk_paints)
            prim.GetAttribute("info:mdl:sourceAsset:subIdentifier").Set(
                random_chalk_paint
            )
        else:
            print(f"    Creating new inputs:project_uvw attribute for {prim.GetName()}")
            # Create the attribute as a boolean
            subIdentifier_attr = prim.CreateAttribute(
                "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token
            )
            # Set it to False (so shadows WILL be cast)
            subIdentifier_attr.Set(random_chalk_paint)

        print(
            f"    Successfully set info:mdl:sourceAsset:subIdentifier for {prim.GetName()}"
        )

    except Exception as e:
        print(
            f"    ERROR: Failed to set info:mdl:sourceAsset:subIdentifier for {prim.GetName()}: {e}"
        )
