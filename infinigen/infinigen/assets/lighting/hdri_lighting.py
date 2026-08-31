# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei
import logging
from pathlib import Path

import bpy
import gin
import numpy as np
from numpy.random import uniform

import infinigen
from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.util.random import random_general as rg

logger = logging.getLogger(__name__)

_HDRI_SUFFIXES = {".exr", ".hdr"}


def default_hdri_dir() -> Path:
    return infinigen.repo_root() / "resources" / "hdri"


def list_hdri_files(folder=None) -> list[Path]:
    folder = Path(folder) if folder is not None else default_hdri_dir()
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _HDRI_SUFFIXES
    )


def existing_world_hdri_image() -> bpy.types.Image | None:
    world = bpy.context.scene.world
    if world is None or world.node_tree is None:
        return None
    for node in world.node_tree.nodes:
        if node.bl_idname == Nodes.EnvironmentTexture and getattr(node, "image", None):
            return node.image
    return None


def _mapping_scale(scale) -> tuple[float, float, float]:
    """Turn a gin float / triple / random_general spec into Mapping XYZ scale.

    Environment textures have no metric size; Mapping Scale > 1 repeats the
    panorama so trees and buildings occupy less of a window (look farther).
    A scalar applies uniformly. A 3-tuple is (X, Y, Z).
    """
    sampled = scale
    if isinstance(scale, (tuple, list)) and scale and isinstance(scale[0], str):
        sampled = rg(scale)
    elif not isinstance(scale, (tuple, list)):
        sampled = rg(scale)
    if isinstance(sampled, (int, float, np.floating)):
        s = max(float(sampled), 1e-3)
        return (s, s, s)
    vals = tuple(float(x) for x in sampled)
    if len(vals) == 1:
        s = max(vals[0], 1e-3)
        return (s, s, s)
    if len(vals) != 3:
        raise ValueError(f"hdri_lighting.scale must be a float or XYZ triple, got {scale!r}")
    return tuple(max(v, 1e-3) for v in vals)


@gin.configurable
def hdri_lighting(
    nw: NodeWrangler,
    strength=("uniform", 6.0, 9.0),
    scale=3.0,
    folder=None,
    reuse_existing=True,
    existing_image=None,
):
    image = existing_image
    if image is None and reuse_existing:
        image = existing_world_hdri_image()
    if image is None:
        files = list_hdri_files(folder)
        if not files:
            raise FileNotFoundError(
                f"No .exr/.hdr files in {folder or default_hdri_dir()}. "
                "Run: python tools/download_polyhaven_hdris.py"
            )
        path = files[int(np.random.randint(0, len(files)))]
        image = bpy.data.images.load(filepath=str(path), check_existing=True)
        try:
            image.pack()
        except Exception:
            logger.debug("Could not pack HDRI %s into the blend", path.name)

    scale_vec = _mapping_scale(scale)
    strength_val = float(rg(strength))
    logger.info(
        "HDRI world: %s strength=%.2f scale=%s",
        image.name,
        strength_val,
        scale_vec,
    )

    texture_coord = nw.new_node(Nodes.TextureCoord)
    coord = nw.new_node(
        Nodes.Mapping,
        [texture_coord],
        input_kwargs={
            "Rotation": (0, 0, uniform(np.pi * 2)),
            "Scale": scale_vec,
        },
    )
    texture = nw.new_node(Nodes.EnvironmentTexture, [coord], attrs={"image": image})
    return nw.new_node(
        Nodes.Background, input_kwargs={"Color": texture, "Strength": strength_val}
    )


def add_lighting():
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    nw = NodeWrangler(world.node_tree)
    surface = hdri_lighting(nw)
    nw.new_node(Nodes.WorldOutput, input_kwargs={"Surface": surface})
