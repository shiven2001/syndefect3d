import bpy
import bmesh
import math

# Clear scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# Create plane
bpy.ops.mesh.primitive_plane_add()
plane = bpy.context.active_object
plane.name = "wall_decoration"

# Scale to reasonable size
plane.scale = (0.8, 0.6, 1.0)

# Rotate so back face will be against wall (minimum X direction)
plane.rotation_euler = (0, math.pi / 2, 0)  # 90° around Y-axis

# Apply transforms
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Add very thin thickness
bpy.ops.object.modifier_add(type="SOLIDIFY")
plane.modifiers["Solidify"].thickness = 0.005  # 5mm - very thin
bpy.ops.object.modifier_apply(modifier="Solidify")

# Triangulate
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.quads_convert_to_tris()
bpy.ops.object.mode_set(mode="OBJECT")

print("Thin wall decoration ready for GLB export!")
