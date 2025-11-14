# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""Generate synthetic datasets using infinigen (https://infinigen.org/) generated environments."""

import argparse
import json
import os
from re import X
import yaml
from isaacsim import SimulationApp

# Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
config = {}

# Load default config from YAML file
default_config_path = "config/infinigen_multi_writers_pt.yaml"
if os.path.isfile(default_config_path):
    try:
        with open(default_config_path, "r") as f:
            config = yaml.safe_load(f)
            print(f"[SDG-Infinigen] Loaded default config from {default_config_path}")
    except Exception as e:
        print(f"[SDG-Infinigen] Error loading default config: {e}")
        print(f"[SDG-Infinigen] Will use empty config")
else:
    print(
        f"[SDG-Infinigen] Default config file {default_config_path} not found, will use empty config"
    )

# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    required=False,
    help="Include specific config parameters (json or yaml))",
)
parser.add_argument(
    "--close-on-completion",
    action="store_true",
    help="Ensure the app closes on completion even in debug mode",
)
args, unknown = parser.parse_known_args()
args_config = {}
if args.config and os.path.isfile(args.config):
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            print(
                f"[SDG-Infinigen] Config file {args.config} is not json or yaml, will use default config"
            )
else:
    print(
        f"[SDG-Infinigen] Config file {args.config} does not exist, will use default config"
    )

# Update the default config dict with the external one
config.update(args_config)
simulation_app = SimulationApp(launch_config={"headless": True})

import random
from itertools import cycle
import carb.settings
import infinigen_sdg_utils as infinigen_utils
import numpy as np
import omni.client
import omni.kit
import omni.kit.app
import omni.physx
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.viewports import set_camera_view


# Run the SDG pipeline on the scenarios
def run_sdg(config):
    # Load the config parameters
    env_config = config.get("environments", {})
    env_urls = infinigen_utils.get_usd_paths(
        files=env_config.get("files", []),
        folders=env_config.get("folders", []),
        skip_folder_keywords=[".thumbs"],
    )
    capture_config = config.get("capture", {})
    writers_config = config.get("writers", {})
    distractors_config = config.get("distractors", {})

    # Create a new stage
    print(f"[SDG-Infinigen] Creating a new stage")
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    # Disable capture on play
    rep.orchestrator.set_capture_on_play(False)

    # Disable UJITSO cooking ([Warning] [omni.ujitso] UJITSO : Build storage validation failed)
    carb.settings.get_settings().set("/physics/cooking/ujitsoCollisionCooking", False)
    carb.settings.get_settings().set("/rtx/ecoMode/enabled", False)
    carb.settings.get_settings().set("/rtx/pathtracing/maxBounces", 8)
    carb.settings.get_settings().set("/rtx-transient/dlssg/enabled", False)
    carb.settings.get_settings().set("/rtx/sceneDb/ambientLightIntensity", 2.0)
    carb.settings.get_settings().set("/rtx/indirectDiffuse/scalingFactor", 5.0)
    # rep.settings.set_render_pathtraced(samples_per_pixel=512)

    # Additional memory optimization settings
    # rep.settings.carb_settings("/omni/replicator/backend/writeThreads", 16)
    # rep.settings.carb_settings("/omni/replicator/backend/queueSize", 2000)

    # Fix for OgnSdPostRenderVarToHost warning - use device buffer for better performance
    # rep.settings.carb_settings("/omni/replicator/backend/useDeviceBuffer", True)
    # rep.settings.carb_settings("/omni/replicator/backend/asyncCopy", True)

    # Memory management optimizations
    # rep.settings.carb_settings(
    #    "/omni/replicator/backend/maxMemoryUsage", 0.85
    # )  # Use 85% of GPU memory
    # rep.settings.carb_settings(
    #    "/omni/replicator/backend/compressionLevel", 1
    # )  # Enable compression
    # rep.settings.carb_settings(
    #     "/omni/replicator/backend/memoryPoolSize", 1024
    # )  # Set memory pool size in MB

    # Additional memory optimization settings for RTX 4090
    # rep.settings.carb_settings("/omni/replicator/backend/maxMemoryUsage", 0.8)
    # 0: Disabled, 1: TAA, 2: FXAA, 3: DLSS, 4:RTXAA
    # carb.settings.get_settings().set("/rtx/post/aa/op", 2)
    # Enable compression
    # rep.settings.carb_settings("/omni/replicator/backend/compressionLevel", 1)
    # carb.settings.get_settings().set("/rtx/post/dlss/execMode", 3)
    # carb.settings.get_settings().set("/rtx-transient/resourcemanager/texturestreaming/memoryBudget", 0.6)

    # Debug mode (hide ceiling, move viewport camera to the top-down view)
    debug_mode = config.get("debug_mode", False)

    # Load the defect materials to the stage
    infinigen_utils.load_defect_materials_to_stage()

    # Load the apartment materials to the stage
    """wood_material_paths, paint_plaster_material_paths, glass_material_paths = (
        infinigen_utils.load_apartment_materials_to_stage()
    )"""

    # Get defect material paths
    defect_material_paths = (
        infinigen_utils.get_defect_material_paths()
    )  # Dynamically find all MDL files in the folder

    ########################################################################################

    # Load the shape distractors
    shape_distractors_config = distractors_config.get("shape_distractors", {})
    floating_shapes, falling_shapes = infinigen_utils.load_shape_distractors(
        shape_distractors_config
    )
    print(f"[SDG-Infinigen] Loaded {len(floating_shapes)} floating shape distractors")
    print(f"[SDG-Infinigen] Loaded {len(falling_shapes)} falling shape distractors")
    shape_distractors = floating_shapes + falling_shapes

    # Load the mesh distractors
    mesh_distractors_config = distractors_config.get("mesh_distractors", {})
    floating_meshes, falling_meshes = infinigen_utils.load_mesh_distractors(
        mesh_distractors_config
    )
    print(f"[SDG-Infinigen] Loaded {len(floating_meshes)} floating mesh distractors")
    print(f"[SDG-Infinigen] Loaded {len(falling_meshes)} falling mesh distractors")
    mesh_distractors = floating_meshes + falling_meshes

    ########################################################################################

    # Resolve any centimeter-meter scale issues of the assets
    infinigen_utils.resolve_scale_issues_with_metrics_assembler()

    # Create lights to randomize in the working area
    scene_lights = []
    num_scene_lights = capture_config.get("num_scene_lights", 0)
    for i in range(num_scene_lights):
        light_prim = stage.DefinePrim(f"/Lights/SphereLight_scene_{i}", "SphereLight")
        scene_lights.append(light_prim)
    print(f"[SDG-Infinigen] Created {len(scene_lights)} scene lights")

    # Register replicator randomizers and trigger them once
    print(f"[SDG-Infinigen] Registering replicator graph randomizers")
    infinigen_utils.register_dome_light_randomizer()
    infinigen_utils.register_shape_distractors_color_randomizer(shape_distractors)

    # Check if the render mode needs to be switched to path tracing for the capture (by default: RayTracedLighting)
    use_path_tracing = capture_config.get("path_tracing", False)

    # Capture detail using subframes (https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html)
    rt_subframes = capture_config.get("rt_subframes", 3)

    # Number of captures (frames = total_captures * num_cameras which are in the stage)
    # NOTE: if captured frames have no labeled data, they can be skipped (e.g. PoseWriter with skip_empty_frames=True)
    total_captures = capture_config.get("total_captures", 0)

    # Number of captures per environment with the objects in the air or dropped
    num_captures_per_env = capture_config.get("num_captures_per_env", 0)
    ##############################################################################
    # Start the SDG loop
    env_cycle = cycle(env_urls)
    capture_counter = 0
    while capture_counter < total_captures:
        # Load the next environment
        env_url = next(env_cycle)
        # Load the new environment
        print(f"[SDG-Infinigen] Loading environment: {env_url}")
        # Clear replicator state to prevent contamination from previous environments
        infinigen_utils.load_env(env_url, prim_path="/Environment")

        # Setup the environment (add collision, fix lights, etc.) and update the app once to apply the changes
        print(f"[SDG-Infinigen] Setting up the environment")
        infinigen_utils.setup_env(root_path="/Environment", hide_top_walls=debug_mode)
        simulation_app.update()

        # Split the geomSubsets into separate meshes# Split geomSubsets into separate meshes
        print("Splitting geomSubsets into separate meshes...")
        infinigen_utils.split_geom_subsets_into_meshes()

        infinigen_utils.set_window_glass_shadow_attributes()

        # Add lighting for the windows
        infinigen_utils.add_sun_lights_for_scene()

        infinigen_utils.turn_off_env_light()

        infinigen_utils.register_randomize_custom_textures_to_scene(
            # wood_material_paths, paint_plaster_material_paths, glass_material_paths
        )
        rep.utils.send_og_event(event_name="randomize_custom_textures_to_scene")

        # Get the location of the prim at which the assets will be randomized
        working_area_loc = infinigen_utils.get_matching_prim_location(
            match_string="dining_room_0_0_wall", root_path="/Environment"
        )

        floating_planes, wall_defects_mesh = (
            infinigen_utils.get_planes_with_defect_materials()
        )

        print(
            f"[SDG-Infinigen] Got {len(floating_planes)} floating defect planes in the scene"
        )

        print(f"[SDG-Infinigen] Registering replicator materials randomizers")
        infinigen_utils.register_randomize_wall_defects_textures(
            wall_defects_mesh, defect_material_paths
        )
        rep.utils.send_og_event(event_name="randomize_defect_texture")

        infinigen_utils.set_point_lamp_intensities(
            give_intensity=30000, give_color_temperature=6500
        )

        # infinigen_utils.set_floor_and_wall_shadow_attributes()

        # Randomize the lights addition by replicator
        # infinigen_utils.register_lights_addition(root_path="/Environment")
        # rep.utils.send_og_event(event_name="lights_addition")

        cameras = infinigen_utils.get_cameras_from_stage()
        print(
            f"[SDG-Infinigen] Got {len(cameras)} cameras"
        )  # Create the render products for the cameras
        render_products = []
        resolution = capture_config.get("resolution", (640, 640))
        disable_render_products = capture_config.get("disable_render_products", False)

        # Extract environment name from the URL for unique naming
        env_name = os.path.basename(env_url).replace(".usd", "").replace(".usdc", "")

        for cam in cameras:
            # Create unique render product name with environment and capture counter
            unique_name = f"rp_{env_name}_{capture_counter:04d}_{cam.GetName()}"
            rp = rep.create.render_product(cam.GetPath(), resolution, name=unique_name)
            if disable_render_products:
                rp.hydra_texture.set_updates_enabled(False)
            render_products.append(rp)
        print(
            f"[SDG-Infinigen] Created {len(render_products)} render products with unique names"
        )

        # Only create the writers if there are render products to attach to
        writers = []
        if render_products:
            for writer_config in writers_config:
                writer = infinigen_utils.setup_writer(writer_config)
                if writer:
                    writer.attach(render_products)
                    writers.append(writer)
                    print(
                        f"\t {writer_config['type']}'s out dir: {writer_config.get('kwargs', {}).get('output_dir', '')}"
                    )
        print(f"[SDG-Infinigen] Created {len(writers)} writers")

        # Move viewport above the working area to get a top-down view of the scene
        if debug_mode:
            camera_loc = (
                working_area_loc[0],
                working_area_loc[1],
                working_area_loc[2] + 20,
            )
            set_camera_view(eye=np.array(camera_loc), target=np.array(working_area_loc))

        ###############################################################################################

        # Randomize the mesh distractors
        print(
            f"\tRandomizing {len(mesh_distractors)} mesh distractors around the working area"
        )
        mesh_loc_range = infinigen_utils.offset_range(
            (-1, -1, 1, 1, 1, 2), working_area_loc
        )
        infinigen_utils.randomize_poses(
            mesh_distractors,
            location_range=mesh_loc_range,
            rotation_range=(0, 360),
            scale_range=(0.3, 1.0),
        )

        # Shape distractors
        print(
            f"\tRandomizing {len(shape_distractors)} shape distractors around the working area"
        )
        shape_loc_range = infinigen_utils.offset_range(
            (-1.5, -1.5, 1, 1.5, 1.5, 2), working_area_loc
        )
        infinigen_utils.randomize_poses(
            shape_distractors,
            location_range=shape_loc_range,
            rotation_range=(0, 360),
            scale_range=(0.01, 0.1),
        )

        ###############################################################################################

        """print(
            f"\tRandomizing {len(scene_lights)} scene lights properties and locations around the working area"
        )
        lights_loc_range = infinigen_utils.offset_range(
            (-2, -2, 1, 2, 2, 3), working_area_loc
        )
        infinigen_utils.randomize_lights(
            scene_lights,
            location_range=lights_loc_range,
            intensity_range=(500, 2500),
            color_range=(0.1, 0.1, 0.1, 0.9, 0.9, 0.9),
        )"""

        infinigen_utils.set_specific_camera_horizontal_aperture()

        print(f"\tRandomizing dome lights")
        rep.utils.send_og_event(event_name="randomize_dome_lights")

        print(f"\tRandomizing shape distractor colors")
        rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")

        # Run the physics simulation for a few frames to solve any collisions
        print(f"\tFixing collisions through physics simulation")
        simulation_app.update()
        infinigen_utils.run_simulation(num_frames=200, render=True)

        # Check if the render products need to be enabled for the capture
        if disable_render_products:
            for rp in render_products:
                rp.hydra_texture.set_updates_enabled(True)

        # Check if the render mode needs to be switched to path tracing for the capture (true by default)
        if use_path_tracing:
            print(f"\tSwitching to PathTracing render mode")
            carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")

        defect_randomizations = 3  # Number of defect randomizations
        texture_randomizations = 3
        light_randomizations = 3

        # Capture frames for this camera with nested randomizations
        for defect_idx in range(defect_randomizations):
            print(f"\t  Defect randomization {defect_idx+1}/{defect_randomizations}")

            # Randomize defects
            print(f"\t    Randomizing defects")
            rep.utils.send_og_event(event_name="randomize_defect_texture")

            for texture_idx in range(texture_randomizations):
                print(
                    f"\t    Texture randomization {texture_idx+1}/{texture_randomizations}"
                )

                # Randomize textures
                print(f"\t      Randomizing textures")
                rep.utils.send_og_event(event_name="randomize_custom_textures_to_scene")
                infinigen_utils.paint_randomizer()

                for light_idx in range(light_randomizations):
                    print(
                        f"\t      Light randomization {light_idx+1}/{light_randomizations}"
                    )
                    infinigen_utils.set_point_lamp_intensities(
                        give_intensity=random.randint(30000, 50000),
                        give_color_temperature=random.randint(4000, 8000),
                    )

                    # Check if the total captures have been reached
                    if capture_counter >= total_captures:
                        break

                    print(
                        f"\t      Capturing image {capture_counter+1} (defect {defect_idx+1}, texture {texture_idx+1})"
                    )
                    rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=0.0)
                    capture_counter += 1

                # Check if we've reached the total captures limit
                if capture_counter >= total_captures:
                    break

                # Check if we've reached the total captures limit
                if capture_counter >= total_captures:
                    break

        if capture_counter >= total_captures:
            break

        ###############################################################################################

        # Check if the render products need to be disabled until the next capture
        if disable_render_products:
            for rp in render_products:
                rp.hydra_texture.set_updates_enabled(False)

        # Check if the render mode needs to be switched back to raytracing until the next capture
        if use_path_tracing:
            carb.settings.get_settings().set("/rtx/rendermode", "RayTracedLighting")

        print(f"\tRunning the simulation")
        infinigen_utils.run_simulation(num_frames=200, render=False)

        # Detach the writers
        print(f"[SDG-Infinigen] Detaching writers")
        for writer in writers:
            writer.detach()

        # Destroy render products
        print(f"[SDG-Infinigen] Destroying render products")
        for rp in render_products:
            rp.destroy()

    # Wait until the data is written to the disk
    rep.orchestrator.wait_until_complete()


# Check if debug mode is enabled
debug_mode = config.get("debug_mode", False)

if debug_mode:
    np.random.seed(10)
    random.seed(10)
    rep.set_global_seed(10)

# Start the SDG pipeline
print(f"[SDG-Infinigen] Starting the SDG pipeline.")
run_sdg(config)
print(f"[SDG-Infinigen] SDG pipeline finished.")

# Make sure the app closes on completion even if in debug mode
if args.close_on_completion:
    simulation_app.close()

# In debug mode, keep the app running until manually closed
if debug_mode:
    while simulation_app.is_running():
        simulation_app.update()

simulation_app.close()
