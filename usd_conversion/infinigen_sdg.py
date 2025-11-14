# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""Convert Infinigen environments to USD format."""

import argparse
import json
import os
from re import X
import yaml
from isaacsim import SimulationApp
import pathlib

Path = pathlib.Path

# Default config dict, can be updated/replaced using json/yaml config files
config_file = {}

script_dir = Path(__file__).parent

# Load default config from YAML file
default_config_path = script_dir / "config.yaml"
if os.path.isfile(default_config_path):
    try:
        with open(default_config_path, "r") as f:
            config_file = yaml.safe_load(f)
            print(f"[USD-Conversion] Loaded default config from {default_config_path}")
    except Exception as e:
        print(f"[USD-Conversion] Error loading default config: {e}")
        print(f"[USD-Conversion] Will use empty config")
else:
    print(
        f"[USD-Conversion] Default config file {default_config_path} not found, will use empty config"
    )

# Update the default config dict with the external one
config = config_file

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


# Run the USD conversion pipeline on the scenarios
def run_usd_conversion(config):
    # Load the config parameters
    env_config = config.get("environments", {})
    env_urls = infinigen_utils.get_usd_paths(
        files=env_config.get("files", []),
        folders=env_config.get("folders", []),
        skip_folder_keywords=[".thumbs"],
    )
    conversion_info = config.get("info", {})

    # Create a new stage
    print(f"[USD-Conversion] Creating a new stage")
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

    ########################################################################################

    # Resolve any centimeter-meter scale issues of the assets
    infinigen_utils.resolve_scale_issues_with_metrics_assembler()

    # Create lights to randomize in the working area
    scene_lights = []
    num_scene_lights = capture_config.get("num_scene_lights", 0)
    for i in range(num_scene_lights):
        light_prim = stage.DefinePrim(f"/Lights/SphereLight_scene_{i}", "SphereLight")
        scene_lights.append(light_prim)
    print(f"[USD-Conversion] Created {len(scene_lights)} scene lights")

    # Register replicator randomizers and trigger them once
    print(f"[USD-Conversion] Registering replicator graph randomizers")
    infinigen_utils.register_dome_light_randomizer()
    infinigen_utils.register_shape_distractors_color_randomizer(shape_distractors)

    # Start the USD conversion loop
    env_cycle = cycle(env_urls)
    capture_counter = 0
    while capture_counter < total_captures:
        # Load the next environment
        env_url = next(env_cycle)
        # Load the new environment
        print(f"[USD-Conversion] Loading environment: {env_url}")
        # Clear replicator state to prevent contamination from previous environments
        infinigen_utils.load_env(env_url, prim_path="/Environment")

        # Setup the environment (add collision, fix lights, etc.) and update the app once to apply the changes
        print(f"[USD-Conversion] Setting up the environment")
        infinigen_utils.setup_env(root_path="/Environment", hide_top_walls=debug_mode)
        simulation_app.update()

        # Split the geomSubsets into separate meshes
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
            f"[USD-Conversion] Got {len(floating_planes)} floating defect planes in the scene"
        )

        print(f"[USD-Conversion] Registering replicator materials randomizers")
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

        # Extract environment name from the URL for unique naming
        env_name = os.path.basename(env_url).replace(".usd", "").replace(".usdc", "")

        ###############################################################################################

        ###############################################################################################

        infinigen_utils.set_specific_camera_horizontal_aperture()

        print(f"\tRandomizing dome lights")
        rep.utils.send_og_event(event_name="randomize_dome_lights")

        # Run the physics simulation for a few frames to solve any collisions
        print(f"\tFixing collisions through physics simulation")
        simulation_app.update()
        infinigen_utils.run_simulation(num_frames=200, render=True)

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

        print(f"\tRunning the simulation")
        infinigen_utils.run_simulation(num_frames=200, render=False)

        # Detach the writers
        print(f"[USD-Conversion] Detaching writers")
        for writer in writers:
            writer.detach()

        # Destroy render products
        print(f"[USD-Conversion] Destroying render products")
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

# Start the USD conversion pipeline
print(f"[USD-Conversion] Starting the USD conversion pipeline.")
run_usd_conversion(config)
print(f"[USD-Conversion] USD conversion pipeline finished.")

# Make sure the app closes on completion even if in debug mode
if args.close_on_completion:
    simulation_app.close()

# In debug mode, keep the app running until manually closed
if debug_mode:
    while simulation_app.is_running():
        simulation_app.update()

simulation_app.close()
