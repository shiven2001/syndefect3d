# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan

import os
import random

import bpy
import numpy as np
from mathutils import Vector

from infinigen.assets.static_assets.base import StaticAssetFactory
from infinigen.core.tagging import tag_support_surfaces
from infinigen.core.util.math import FixedSeed, int_hash


def static_category_factory(
    path_to_assets: str,
    tag_support=False,
    x_dim: float = None,
    y_dim: float = None,
    z_dim: float = None,
    rotation_euler: tuple[float] = None,
) -> StaticAssetFactory:
    """
    Create a factory for external asset import.
    tag_support: tag the planes of the object that are parallel to xy plane as support surfaces (e.g. shelves)
    x_dim, y_dim, z_dim: specify ONLY ONE dimension for the imported object. The object will be scaled accordingly.
    rotation_euler: sets the rotation of the object in euler angles. The object will not be rotated if not specified.
    """

    class StaticCategoryFactory(StaticAssetFactory):
        def __init__(self, factory_seed, coarse=False):
            super().__init__(factory_seed, coarse)
            with FixedSeed(factory_seed):
                self.path_to_assets = path_to_assets
                self.tag_support = tag_support
                self.asset_dir = path_to_assets
                self.x_dim, self.y_dim, self.z_dim = x_dim, y_dim, z_dim
                self.rotation_euler = rotation_euler
                # Cache the list of available asset files
                self.asset_files = [
                    f
                    for f in os.listdir(self.asset_dir)
                    if f.lower().endswith(tuple(self.import_map.keys()))
                ]
                if not self.asset_files or len(self.asset_files) == 0:
                    raise ValueError(f"No valid asset files found in {self.asset_dir}")

        def create_asset(self, i=None, **params) -> bpy.types.Object:
            # Select a random asset for this spawn using the spawn index i
            # Use a combination of factory_seed and i for reproducible random selection
            if i is not None:
                with FixedSeed(int_hash((self.factory_seed, i))):
                    asset_file = random.choice(self.asset_files)
            else:
                asset_file = random.choice(self.asset_files)

            # Store the selected asset file for use in naming
            self.asset_file = asset_file

            file_path = os.path.join(self.asset_dir, asset_file)
            imported_obj = self.import_file(file_path)
            if (
                self.x_dim is not None
                or self.y_dim is not None
                or self.z_dim is not None
            ):
                # check only one dimension is provided
                if (
                    sum(
                        [
                            1
                            for dim in [self.x_dim, self.y_dim, self.z_dim]
                            if dim is not None
                        ]
                    )
                    != 1
                ):
                    raise ValueError("Only one dimension can be provided")
                if self.x_dim is not None:
                    scale = self.x_dim / imported_obj.dimensions[0]
                elif self.y_dim is not None:
                    scale = self.y_dim / imported_obj.dimensions[1]
                else:
                    scale = self.z_dim / imported_obj.dimensions[2]
                imported_obj.scale = (scale, scale, scale)
            if self.tag_support:
                tag_support_surfaces(imported_obj)

            if imported_obj:
                return imported_obj
            else:
                raise ValueError(f"Failed to import asset: {self.asset_file}")

        def spawn_asset(self, i, **kwargs):
            # Call parent's spawn_asset to handle the asset creation
            result = super().spawn_asset(i, **kwargs)

            # Handle export case where result is a tuple
            if isinstance(result, tuple):
                obj, export_path, semantic_mapping = result
            else:
                obj = result
                export_path = None
                semantic_mapping = None

            # Extract asset filename without extension and use it as the object name
            # This makes it easy to identify which asset was used for each spawned object
            asset_name = os.path.splitext(self.asset_file)[0]

            # Use just the asset name as the object name
            # Blender has a 63 character limit on object names, so truncate if needed
            max_name_length = 63
            if len(asset_name) > max_name_length:
                asset_name = asset_name[:max_name_length]

            obj.name = asset_name

            # Return in the same format as parent
            if export_path is not None:
                return obj, export_path, semantic_mapping
            return obj

        def finalize_assets(self, assets):
            """
            Post-process defect planes to embed them into the wall.
            Moves defect planes -0.00226m inward from their current position.
            """
            from infinigen.core import tags as t
            from infinigen.core.util import blender as butil
            import numpy as np

            # Check if this is a defect plane factory by checking the path
            is_defect_plane = "DefectPlane" in self.path_to_assets

            if not is_defect_plane:
                # For non-defect planes, just call parent's finalize_assets
                return (
                    super().finalize_assets(assets)
                    if hasattr(super(), "finalize_assets")
                    else None
                )

            # Embed defect planes into the wall by moving them inward
            # Current position: 1mm out from wall (0.001m margin)
            # Desired position: -0.00126m (1.26mm into the wall)
            # Movement needed: -0.00226m inward
            EMBED_OFFSET = -0.00226  # Move 2.26mm inward into the wall

            for obj in assets:
                if obj.type != "MESH" or not obj.data.polygons:
                    continue

                # Get the back face normal (defect planes attach via back face to wall)
                try:
                    # Try to get back face from tags
                    from infinigen.core.tagging import tagged_face_mask

                    back_mask = tagged_face_mask(obj, {t.Subpart.Back})

                    if back_mask.any():
                        # Use the largest back face polygon
                        back_faces = [i for i, tag in enumerate(back_mask) if tag]
                        if back_faces:
                            largest_back_face_idx = max(
                                back_faces, key=lambda idx: obj.data.polygons[idx].area
                            )
                            back_poly = obj.data.polygons[largest_back_face_idx]
                        else:
                            continue
                    else:
                        # Fallback: find face with most negative Y normal component
                        # (assuming back face points in -Y direction typically)
                        back_poly = max(
                            obj.data.polygons,
                            key=lambda p: -p.normal.y if p.normal.y < 0 else -1e6,
                        )

                    # Get the global wall normal
                    # The back face normal points from the defect plane away from the wall
                    # We want to move along the back face normal direction (into the wall)
                    wall_normal = np.array(butil.global_polygon_normal(obj, back_poly))

                    # Move the object inward along the wall normal
                    # EMBED_OFFSET is negative, so this moves into the wall
                    translation = Vector(wall_normal * EMBED_OFFSET)
                    obj.location += translation

                except Exception as e:
                    # If anything fails, log and continue
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to embed defect plane {obj.name}: {e}")
                    continue

            # Call parent's finalize_assets if it exists
            if hasattr(super(), "finalize_assets"):
                return super().finalize_assets(assets)

    return StaticCategoryFactory


# Create factory instances for different categories
StaticSofaFactory = static_category_factory(
    "infinigen/assets/static_assets/source/Sofa"
)
StaticTableFactory = static_category_factory(
    "infinigen/assets/static_assets/source/Table"
)
StaticShelfFactory = static_category_factory(
    "infinigen/assets/static_assets/source/Shelf", tag_support=True, z_dim=2
)
StaticDefectPlaneFactory = static_category_factory(
    "infinigen/assets/static_assets/source/DefectPlane", z_dim=1
)
StaticACFactory = static_category_factory(
    "infinigen/assets/static_assets/source/AC",
    z_dim=0.5,  # 40cm tall (realistic AC unit size)
)
StaticWallPlugFactory = static_category_factory(
    "infinigen/assets/static_assets/source/Plugs"
)

# Faucets (wall-mounted, e.g. in bathrooms); assumes assets are under source/Faucets
StaticFaucetFactory = static_category_factory(
    "infinigen/assets/static_assets/source/Faucets"
)
