# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Zeyu Ma, Kaiyu Yang, Lingjie Mei


import logging
import math

import bpy
import gin
import numpy as np
from numpy.random import uniform

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.math import clip_gaussian
from infinigen.core.util.random import random_general as rg

logger = logging.getLogger(__name__)


@gin.configurable
def nishita_lighting(
    nw,
    cam,
    dust_density=("clip_gaussian", 1, 1, 0.1, 2),
    air_density=("clip_gaussian", 1, 0.2, 0.7, 1.3),
    strength=("uniform", 0.18, 0.22),
    sun_intensity=("uniform", 0.8, 1),
    sun_elevation=("spherical_sample", 10, None),
    sun_size_deg=("clip_gaussian", 0.5, 0.3, 0.25, 5),
    dynamic=False,
    rising_angle=90,
    camera_based_rotation=None,
):
    sky_texture = nw.new_node(Nodes.SkyTexture)
    sky_texture.sky_type = "NISHITA"
    sky_texture.sun_size = np.deg2rad(rg(sun_size_deg))
    sky_texture.sun_intensity = rg(sun_intensity)
    sky_texture.sun_elevation = np.radians(rg(sun_elevation))
    if camera_based_rotation is None:
        sky_texture.sun_rotation = np.random.uniform(0, 2 * math.pi)
    else:
        sky_texture.sun_rotation = (
            2 * math.pi
            - cam.parent.rotation_euler[2]
            + np.radians(camera_based_rotation)
        )
    if dynamic:
        sky_texture.sun_rotation += (
            (sky_texture.sun_elevation + np.radians(8))
            / 2
            * np.arctan(np.radians(rising_angle))
        )
        sky_texture.keyframe_insert(
            data_path="sun_rotation", frame=bpy.context.scene.frame_end
        )
        sky_texture.sun_rotation -= (
            sky_texture.sun_elevation + np.radians(8)
        ) * np.arctan(np.radians(rising_angle))
        sky_texture.keyframe_insert(
            data_path="sun_rotation", frame=bpy.context.scene.frame_start
        )

        sky_texture.keyframe_insert(
            data_path="sun_elevation", frame=bpy.context.scene.frame_end
        )
        sky_texture.sun_elevation = -np.radians(8)
        sky_texture.keyframe_insert(
            data_path="sun_elevation", frame=bpy.context.scene.frame_start
        )
        sky_texture.sun_elevation = -np.radians(5)
        sky_texture.keyframe_insert(
            data_path="sun_elevation", frame=bpy.context.scene.frame_start + 10
        )

    sky_texture.altitude = clip_gaussian(100, 400, 0, 2000)
    sky_texture.air_density = rg(air_density)
    sky_texture.dust_density = rg(dust_density)
    sky_texture.ozone_density = clip_gaussian(1, 1, 0.1, 10)

    strength = rg(strength)
    return nw.new_node(
        Nodes.Background, input_kwargs={"Color": sky_texture, "Strength": strength}
    )


def _reset_world_tree():
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    return world


@gin.configurable
def add_lighting(cam=None, mode="nishita"):
    """Install the world shader. ``mode`` is ``nishita`` (procedural sky) or ``hdri``.

    HDRI uses ``hdri_lighting`` and files in ``resources/hdri``. If that folder is
    empty, falls back to Nishita. An HDRI already packed in the .blend is reused
    as the image, but Mapping scale / strength are always rebuilt so gin
    overrides apply at render.
    """
    from infinigen.assets.lighting import hdri_lighting as hdri_mod

    use_hdri = mode == "hdri"
    cached_image = hdri_mod.existing_world_hdri_image() if use_hdri else None
    if use_hdri and cached_image is None and not hdri_mod.list_hdri_files():
        logger.warning(
            "add_lighting mode=hdri but resources/hdri has no .exr/.hdr; "
            "falling back to Nishita. Run python tools/download_polyhaven_hdris.py"
        )
        use_hdri = False

    _reset_world_tree()
    nw = NodeWrangler(bpy.context.scene.world.node_tree)
    if use_hdri:
        surface = hdri_mod.hdri_lighting(nw, existing_image=cached_image)
    else:
        surface = nishita_lighting(nw, cam)

    nw.new_node(Nodes.WorldOutput, input_kwargs={"Surface": surface, "Volume": None})


@gin.configurable
def add_multi_directional_sun_lighting(
    n_directions: int = 2,
    elevation_deg: float = 40.0,
    energy_per_sun: float = 2.0,
    sun_angle_deg: float = 0.53,
):
    """
    Add sun lamps from multiple directions so light shines through all sides
    of the apartment (e.g., through windows on different walls).
    """
    elevation_rad = math.radians(elevation_deg)
    for i in range(n_directions):
        azimuth_rad = 2 * math.pi * i / n_directions
        # Sun lamp default points -Z (light travels in -Z). We need light from above, pointing DOWN.
        # rot_x = pi/2 - elevation: (0,0,-1) -> (0, cos(el), -sin(el)) so Z is negative (downward).
        # rot_z = azimuth rotates the horizontal component to the correct compass direction.
        rot_x = math.pi / 2 - elevation_rad
        rot_z = azimuth_rad
        bpy.ops.object.light_add(type="SUN", location=(0, 0, 0))
        sun = bpy.context.active_object
        sun.name = f"SunMulti_{i}"
        sun.rotation_euler = (rot_x, 0, rot_z)
        sun.data.energy = energy_per_sun
        sun.data.angle = math.radians(sun_angle_deg)


@gin.configurable
def add_camera_based_lighting(
    enabled=False,
    energy=("log_uniform", 200, 500),
    spot_size=("uniform", np.pi / 6, np.pi / 4),
):
    """Optional inspection / phone-flash spot from the active camera.

    Disabled by default so existing indoor lighting is unchanged. Enable via gin
    (``add_camera_based_lighting.enabled = True``) for close-up defect renders.
    """
    if not enabled:
        return None
    camera = bpy.context.scene.camera
    if camera is None:
        return None
    bpy.ops.object.light_add(
        type="SPOT", location=camera.location, rotation=camera.rotation_euler
    )
    spot = bpy.context.active_object
    spot.name = "InspectionSpot"
    spot.data.energy = rg(energy)
    spot.data.spot_size = rg(spot_size)
    spot.data.spot_blend = uniform(0.6, 0.8)
    return spot
