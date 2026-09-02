# Grout / plank-joint chips on tiled walls and floors.
#
# Tile joints live in the host shader (Brick / hex), not as mesh edges, so
# placement samples the recorded layout from ceramic.tile / TiledWood and
# seats a small dense decal on those lines. Intact tile is transparent;
# only the bite carries TileChipMaterial for material-index export.
#
# Two hard-won constraints shape everything below:
#
# * The decal may NOT displace itself into the host. The host wall/floor is
#   opaque and its own shader displacement is BUMP-only, so its surface never
#   moves; anything pushed behind that plane is simply occluded. A chip is a
#   recess, but it has to be *shaded* as one from a card sitting a hair proud
#   (same trick as edge_chip_plane), never modelled as one.
# * A chip reads as damage only if it is close to the colour it came out of,
#   but still a stop away from the glaze: ceramic biscuit is desaturated and
#   darker on a light tile, paler on a dark one. Sampling the host keeps the
#   hue honest without blending into the tile.

from __future__ import annotations

import logging
import math
from math import sqrt

import bmesh
import bpy
import gin
import numpy as np
from mathutils import Euler, Matrix, Vector

from infinigen.assets.crack_plane import _mean_material_rgb
from infinigen.assets.materials.ceramic.tile_layout import layout_from_object
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, int_hash

logger = logging.getLogger(__name__)

MATERIAL_PREFIX = "TileChipMaterial"

# Clear of the host face by more than the render-time bevel and any shader
# bump, but under a pixel at close-up framing (~0.6 mm at 0.4 m).
PROUD_M = 0.0006

# Real chips: a few mm of glaze off a grout line up to a ~2.5 cm corner bite;
# laminate loses larger flakes of wear layer than ceramic loses glaze.
# The previous 4–14 mm ceramic band sat on ~8 px of a room camera and read
# as absent even when the decal was seated correctly.
CHIP_MIN_M = 0.008
CHIP_MAX_M = 0.026
CHIP_MAX_WOOD_M = 0.040

# Canvas around the bite. Sized to the chip so the defect-focus camera frames
# the damage and not a mostly-empty card.
PATCH_RATIO = 3.2
PATCH_MIN_M = 0.028
PATCH_MAX_M = 0.120
PATCH_CUTS = 16

# Air required in front of a candidate site. Wall shells have room-side and
# cavity-side faces with unreliable winding, so the open side is measured, not
# assumed. 0.55 m rejected every site in a tight bathroom; 0.22 m still
# excludes the wall cavity (~0.2 m) while allowing chips where a close-up
# camera can stand.
MIN_CLEARANCE_M = 0.22
MAX_CLEARANCE_M = 0.90

# TiledWood renders far lighter than the base colour it is handed: measured
# across its palette (base luminance 0.02 - 0.66), the albedo the shader
# effectively presents is ~0.045 + 0.268*sqrt(base_lum), so even a near-black
# board still reads around 0.07. Anchoring the exposed core to base_color
# directly therefore blew out to cream on dark planks.
WOOD_EFF_ALBEDO = (0.045, 0.268)

HEX_AXES = np.array(
    [
        [1.0 / sqrt(3.0), -1.0 / 3.0],
        [0.0, 2.0 / 3.0],
        [-1.0 / sqrt(3.0), -1.0 / 3.0],
    ],
    dtype=float,
)


def _generation_seed():
    try:
        return int(gin.query_parameter("OVERALL_SEED"))
    except Exception:
        return 0


def _disable_chip_lighting(obj: bpy.types.Object) -> None:
    obj.visible_shadow = False
    obj.visible_diffuse = False
    if hasattr(obj, "visible_glossy"):
        obj.visible_glossy = False
    if hasattr(obj, "visible_transmission"):
        obj.visible_transmission = False
    if hasattr(obj, "visible_volume_scatter"):
        obj.visible_volume_scatter = False


def _room_is_kept(name, keep_rooms):
    if keep_rooms is None:
        return True
    return any(k.split(".")[0] in name for k in keep_rooms)


def _as_vec(val, default=(0.0, 0.0, 0.0)) -> Vector:
    if val is None:
        return Vector(default)
    return Vector((float(val[0]), float(val[1]), float(val[2])))


def _apply_mapping(vec: Vector, layout: dict) -> Vector:
    loc = _as_vec(layout.get("loc"))
    rot = _as_vec(layout.get("rot"))
    scale = _as_vec(layout.get("scale"), (1.0, 1.0, 1.0))
    v = Vector((vec.x * scale.x, vec.y * scale.y, vec.z * scale.z))
    v = Euler((rot.x, rot.y, rot.z), "XYZ").to_matrix() @ v
    return v + loc


def _to_tile_plane(p: Vector, n: Vector, vertical: bool) -> Vector:
    if n.length > 1e-8:
        n = n.normalized()
    if not vertical:
        return Vector((p.x, p.y, 0.0))
    c = p.cross(n)
    return Vector((c.z, p.z, 0.0))


def _mapped(p_obj: Vector, n_obj: Vector, layout: dict) -> Vector:
    plane = _to_tile_plane(p_obj, n_obj, bool(layout.get("vertical")))
    mapped = _apply_mapping(plane, layout)
    y_shift = float(layout.get("y_shift") or 0.0)
    if y_shift:
        mapped = Vector((mapped.x, mapped.y + y_shift, mapped.z))
    return mapped


def _brick_scale(layout: dict) -> float:
    s = float(layout.get("brick_scale") or 1.0)
    return s if abs(s) > 1e-8 else 1.0


def _mortar_m(layout: dict) -> float:
    """Grout width in metres.

    Brick layouts express mortar in the texture units the Brick node works in,
    which ``brick_scale`` converts back to metres; the hex layout already
    measures both ``size`` and ``mortar`` in metres.
    """
    mortar = float(layout.get("mortar") or 0.008)
    if str(layout.get("shape") or "square") == "hexagon":
        return float(np.clip(mortar, 0.0008, 0.012))
    return float(np.clip(mortar / _brick_scale(layout), 0.0008, 0.012))


def _brick_local(mapped: Vector, layout: dict):
    """Position inside one brick cell, plus the cell size, in texture units."""
    scale = _brick_scale(layout)
    x = mapped.x * scale
    y = mapped.y * scale
    row_height = float(layout.get("row_height") or 1.0) or 1.0
    brick_width = float(layout.get("brick_width") or 1.0) or 1.0
    rownum = math.floor(y / row_height)
    of = int(layout.get("offset_frequency") or 0)
    sf = int(layout.get("squash_frequency") or 0)
    bw = brick_width
    offset = 0.0
    if of != 0 and sf != 0:
        if rownum % sf == 0:
            bw *= float(layout.get("squash_amount") or 1.0)
        if rownum % of == 0:
            offset = bw * float(layout.get("offset_amount") or 0.0)
    if bw <= 1e-8:
        bw = brick_width
    bricknum = math.floor((x + offset) / bw)
    lx = (x + offset) - bw * bricknum
    ly = y - row_height * rownum
    return lx, ly, bw, row_height


def _hex_signed(mapped: Vector, layout: dict) -> float:
    size = float(layout.get("size") or 0.22) or 0.22
    mortar = float(layout.get("mortar") or 0.008)
    q = HEX_AXES @ np.array([mapped.x, mapped.y], dtype=float) / size
    qr = np.round(q)
    qd = np.abs(q - qr)
    coords = np.zeros(3, dtype=float)
    for i in range(3):
        if qd[i] > qd[(i + 1) % 3] and qd[i] > qd[(i + 2) % 3]:
            coords[i] = qr[i]
        else:
            coords[i] = -(qr[(i + 1) % 3] + qr[(i + 2) % 3])
    diffs = np.abs(q - coords)
    max_dist = float(max(diffs[0] + diffs[1], diffs[1] + diffs[2], diffs[2] + diffs[0]))
    thresh = 1.0 - mortar / size / 2.0
    return thresh - max_dist


def _inplane_basis(n: Vector):
    n = n.normalized()
    helper = Vector((0.0, 0.0, 1.0)) if abs(n.z) < 0.9 else Vector((1.0, 0.0, 0.0))
    t1 = n.cross(helper)
    if t1.length < 1e-6:
        t1 = n.cross(Vector((0.0, 1.0, 0.0)))
    t1.normalize()
    t2 = n.cross(t1).normalized()
    return t1, t2


def _joint_field(p_obj: Vector, n_obj: Vector, layout: dict, eps: float = 0.004):
    """Distance to the nearest tile joint and its in-plane gradient.

    Returns ``(f, grad, du_m, dv_m, vert_grout)`` where ``f`` is zero on a joint
    centreline and ``grad`` is d(f)/d(position) in the face plane, so a Newton
    step ``p -= f * grad / |grad|^2`` lands the sample exactly on the joint
    rather than merely inside a fat band around it.
    """
    t1, t2 = _inplane_basis(n_obj)
    shape = str(layout.get("shape") or "square")

    if shape == "hexagon":
        f0 = _hex_signed(_mapped(p_obj, n_obj, layout), layout)
        f1 = _hex_signed(_mapped(p_obj + t1 * eps, n_obj, layout), layout)
        f2 = _hex_signed(_mapped(p_obj + t2 * eps, n_obj, layout), layout)
        grad = (t1 * (f1 - f0) + t2 * (f2 - f0)) / eps
        # Hex signed value scales with cell size; convert to a rough metre
        # distance for the corner test using |grad|. Only the distance to the
        # nearest edge is cheap to get here, and hex tiles meet three-at-a-time
        # rather than in the square corners the corner form draws, so hex sites
        # are always treated as edge bites.
        d_m = abs(f0) / max(grad.length, 1e-6)
        return f0, grad, d_m, 1e3, False

    m0 = _mapped(p_obj, n_obj, layout)
    ju = (_mapped(p_obj + t1 * eps, n_obj, layout) - m0) / eps
    jv = (_mapped(p_obj + t2 * eps, n_obj, layout) - m0) / eps
    scale = _brick_scale(layout)
    grad_x = (t1 * ju.x + t2 * jv.x) * scale
    grad_y = (t1 * ju.y + t2 * jv.y) * scale

    lx, ly, bw, rh = _brick_local(m0, layout)
    if lx <= bw - lx:
        du, gdu = lx, grad_x
    else:
        du, gdu = bw - lx, -grad_x
    if ly <= rh - ly:
        dv, gdv = ly, grad_y
    else:
        dv, gdv = rh - ly, -grad_y

    vert_grout = du <= dv
    f, grad = (du, gdu) if vert_grout else (dv, gdv)
    inv = 1.0 / scale
    return f, grad, du * inv, dv * inv, vert_grout


def _snap_to_joint(p_obj: Vector, n_obj: Vector, layout: dict, max_step: float):
    """Walk the sample onto the joint centreline it is nearest to."""
    p = p_obj.copy()
    moved = 0.0
    for _ in range(3):
        f, grad, du_m, dv_m, vert = _joint_field(p, n_obj, layout)
        g2 = grad.length_squared
        if g2 < 1e-12:
            return None
        step = grad * (-f / g2)
        if step.length > max_step - moved:
            return None
        moved += step.length
        p = p + step
        if abs(f) / max(grad.length, 1e-6) < 2e-4:
            break
    f, grad, du_m, dv_m, vert = _joint_field(p, n_obj, layout)
    if abs(f) / max(grad.length, 1e-6) > 1e-3:
        return None
    return p, du_m, dv_m, vert


def _reproject(host, p_obj: Vector, n_obj: Vector):
    """Pull the snapped point back onto the host surface, or reject it.

    Snapping moves in the tangent plane, which can walk off the end of the
    face it started on. A short cast back at the mesh both confirms the point
    is still on the host and re-seats it on the exact surface.
    """
    for sign in (1.0, -1.0):
        hit, loc, nrm, _idx = host.ray_cast(
            p_obj + n_obj * (0.03 * sign), n_obj * -sign, distance=0.06
        )
        if hit:
            return loc, (nrm.normalized() if nrm.length > 1e-6 else n_obj)
    return None


def _open_side(depsgraph, p_world: Vector, n_world: Vector):
    """Return the face normal that points into the room, or None.

    Wall shells carry faces whose winding cannot be trusted, and the cavity
    between the interior wall and the exterior shell is only ~0.2 m deep -
    a chip seated there is invisible to every camera in the scene. Measuring
    the free run on both sides rejects the cavity and fixes flipped normals in
    the same test.
    """
    best_d, best_n = -1.0, None
    for d in (n_world, -n_world):
        hit, loc, _n, _i, _o, _m = bpy.context.scene.ray_cast(
            depsgraph, p_world + d * 0.015, d, distance=MAX_CLEARANCE_M
        )
        dist = (loc - p_world).length if hit else MAX_CLEARANCE_M
        if dist > best_d:
            best_d, best_n = dist, d
    if best_d < MIN_CLEARANCE_M:
        return None
    return best_n


def _host_chip_colors(kind: str, host_rgb):
    """Chip body, rim and contact-shadow colours derived from the host finish.

    Sampled from the host rather than invented, so a chip in a white tile
    exposes near-white biscuit instead of the mid grey an independent sample
    lands on. Values here are linear, as Blender colour sockets are.
    """
    if host_rgb is None:
        host_rgb = (0.12, 0.08, 0.05) if kind == "wood" else (0.35, 0.34, 0.32)
    r, g, b = (float(np.clip(c, 0.0, 1.0)) for c in host_rgb[:3])
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    if kind == "wood":
        # Hue from the substrate, value pinned to what the plank actually
        # renders at: a core is clearly lighter than a dark board and close to
        # level with a pale one, never a cream splash on either.
        a0, a1 = WOOD_EFF_ALBEDO
        eff = a0 + a1 * math.sqrt(max(lum, 0.0))
        t = float(np.clip((eff - 0.07) / 0.19, 0.0, 1.0))
        # Band chosen against rendered plank luminance, not the fit alone: the
        # fit runs ~1.4x hot at the dark end of the palette.
        ratio = (1.35 - 0.60 * t) * float(np.random.uniform(0.92, 1.08))
        target = float(np.clip(eff * ratio, 0.04, 0.42))
        core = (0.62, 0.50, 0.34)
        mix = float(np.random.uniform(0.60, 0.85))
        tint = [(1.0 - mix) * c + mix * k for c, k in zip((r, g, b), core)]
        cur = max(0.2126 * tint[0] + 0.7152 * tint[1] + 0.0722 * tint[2], 1e-4)
        base = [c * (target / cur) for c in tint]
    else:
        desat = float(np.random.uniform(0.62, 0.88))
        tinted = [(1.0 - desat) * c + desat * lum for c in (r, g, b)]
        # Glaze is a thin coloured skin over a pale body. On a white bathroom
        # tile a 15% darkening is lost in the grout; on dark slate the biscuit
        # has to go paler or it vanishes into the glaze. Keep both cases, but
        # put a real stop of contrast on light tiles.
        t = float(np.clip((lum - 0.05) / 0.30, 0.0, 1.0))
        ratio = (1.65 - 1.10 * t) * float(np.random.uniform(0.92, 1.08))
        cur = max(
            0.2126 * tinted[0] + 0.7152 * tinted[1] + 0.0722 * tinted[2], 1e-4
        )
        base = [c * (lum * ratio / cur) for c in tinted]

    def rgba(vals, mul):
        return tuple(float(np.clip(c * mul, 0.0, 1.0)) for c in vals) + (1.0,)

    return (
        rgba(base, np.random.uniform(0.58, 0.72)),  # body low
        rgba(base, np.random.uniform(1.02, 1.16)),  # body high
        rgba(base, np.random.uniform(1.22, 1.42)),  # fresh fracture rim
        rgba(base, np.random.uniform(0.16, 0.30)),  # contact shadow at grout
    )


def create_tile_chip_material(
    name: str,
    seed: int,
    kind: str,
    corner: bool,
    patch_m: float,
    chip_m: float,
    grout_m: float,
    host_rgb=None,
) -> bpy.types.Material:
    with FixedSeed(seed):
        lo, hi, rim_col, shadow_col = _host_chip_colors(kind, host_rgb)
        rough = (
            float(np.random.uniform(0.84, 0.95))
            if kind == "wood"
            else float(np.random.uniform(0.76, 0.90))
        )
        # Extents of the bite as it actually shows, in metres. Both forms are
        # clipped at the joint, so the across-joint radius is the whole depth
        # rather than a half-axis - treating it as one gave 1 mm slivers.
        if corner:
            width_m = chip_m * float(np.random.uniform(0.85, 1.25))
            depth_m = width_m * float(np.random.uniform(0.75, 1.25))
        else:
            width_m = chip_m * float(np.random.uniform(1.00, 1.45))
            depth_m = width_m * float(np.random.uniform(0.35, 0.75))
        warp_amp = float(np.random.uniform(0.12, 0.24))
        teeth_amp = float(np.random.uniform(0.06, 0.13))
        warp_cyc = float(np.random.uniform(3.5, 6.5))
        teeth_cyc = float(np.random.uniform(15.0, 28.0))
        grain_cyc = float(np.random.uniform(6.0, 14.0))
        rim_strength = float(np.random.uniform(0.38, 0.62))
        ao_strength = float(np.random.uniform(0.48, 0.72))
        shadow_strength = float(np.random.uniform(0.68, 0.90))
        spec = float(np.random.uniform(0.02, 0.10))
        bump_strength = float(np.random.uniform(0.55, 0.88))
        bump_dist = float(np.random.uniform(0.00070, 0.00180))

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    def node(kind_, **props):
        n = nodes.new(kind_)
        for k, val in props.items():
            setattr(n, k, val)
        return n

    def math_op(op, a, b=None, clamp=False):
        m = node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, operand in enumerate((a, b)):
            if operand is None:
                continue
            if hasattr(operand, "is_linked"):
                links.new(operand, m.inputs[i])
            else:
                m.inputs[i].default_value = operand
        return m.outputs["Value"]

    def ramp(value, x0, x1, y0=0.0, y1=1.0, interp="SMOOTHSTEP"):
        mr = node("ShaderNodeMapRange", interpolation_type=interp, clamp=True)
        mr.inputs["From Min"].default_value = x0
        mr.inputs["From Max"].default_value = x1
        mr.inputs["To Min"].default_value = y0
        mr.inputs["To Max"].default_value = y1
        links.new(value, mr.inputs["Value"])
        return mr.outputs["Result"]

    def noise(vector, scale, detail=8.0, roughness=0.60):
        n = node("ShaderNodeTexNoise", noise_dimensions="3D", noise_type="FBM")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = roughness
        links.new(vector, n.inputs["Vector"])
        return n.outputs["Fac"]

    def signed_noise(vector, scale, detail, roughness, amp):
        """Noise Fac in [-amp, amp].

        An FBM Fac clusters around 0.5 with roughly a +/-0.15 spread, so
        ``(Fac - 0.5) * 2 * amp`` delivers a third of the requested amplitude
        and the chip outline comes out as a smooth lens instead of a fracture.
        """
        return ramp(
            noise(vector, scale, detail, roughness),
            0.35, 0.65, -amp, amp, interp="LINEAR",
        )

    def mix_rgb(fac, c1, c2):
        m = node("ShaderNodeMixRGB", blend_type="MIX")
        if hasattr(fac, "is_linked"):
            links.new(fac, m.inputs["Fac"])
        else:
            m.inputs["Fac"].default_value = fac
        for sock, col in (("Color1", c1), ("Color2", c2)):
            if hasattr(col, "is_linked"):
                links.new(col, m.inputs[sock])
            else:
                m.inputs[sock].default_value = col
        return m.outputs["Color"]

    tex = node("ShaderNodeTexCoord")
    sep = node("ShaderNodeSeparateXYZ")
    links.new(tex.outputs["Generated"], sep.inputs["Vector"])
    u, v = sep.outputs["X"], sep.outputs["Y"]

    # Noise is driven in metres so the jag frequency is independent of how big
    # this particular canvas ended up.
    uvm = node("ShaderNodeCombineXYZ")
    links.new(math_op("MULTIPLY", u, patch_m), uvm.inputs["X"])
    links.new(math_op("MULTIPLY", v, patch_m), uvm.inputs["Y"])
    uv_m = uvm.outputs["Vector"]

    # Grout takes half its width off the patch centre before tile starts.
    grout_uv = float(np.clip(grout_m * 0.5 / max(patch_m, 1e-6), 0.0, 0.08))
    edge0 = 0.5 + grout_uv
    # A corner bite is clipped into one quadrant, so both radii are full
    # extents; the edge lens is symmetric about the joint normal, so only its
    # along-joint radius is a half-width.
    ru = max((width_m if corner else width_m * 0.5) / patch_m, 1e-4)
    rv = max(depth_m / patch_m, 1e-4)

    if corner:
        # Bite out of the tile corner: a disc centred on the joint crossing,
        # clipped to the one tile that lost the material.
        du = math_op("DIVIDE", math_op("SUBTRACT", u, edge0), ru)
        dv = math_op("DIVIDE", math_op("SUBTRACT", v, edge0), rv)
        keep = math_op(
            "MULTIPLY",
            ramp(u, edge0 - 0.004, edge0 + 0.004, 0.0, 1.0),
            ramp(v, edge0 - 0.004, edge0 + 0.004, 0.0, 1.0),
        )
        joint_prox = math_op(
            "MAXIMUM",
            ramp(u, edge0, edge0 + ru * 0.35, 1.0, 0.0),
            ramp(v, edge0, edge0 + rv * 0.35, 1.0, 0.0),
        )
    else:
        # Shallow lens sitting on the joint and biting into one tile.
        du = math_op("DIVIDE", math_op("SUBTRACT", u, 0.5), ru)
        dv = math_op("DIVIDE", math_op("SUBTRACT", v, edge0), rv)
        keep = ramp(v, edge0 - 0.004, edge0 + 0.004, 0.0, 1.0)
        joint_prox = ramp(v, edge0, edge0 + rv * 0.40, 1.0, 0.0)

    dist = math_op(
        "POWER",
        math_op("ADD", math_op("MULTIPLY", du, du), math_op("MULTIPLY", dv, dv)),
        0.5,
    )
    warp = signed_noise(uv_m, warp_cyc / max(chip_m, 1e-4), 4.0, 0.55, warp_amp)
    teeth = signed_noise(uv_m, teeth_cyc / max(chip_m, 1e-4), 9.0, 0.78, teeth_amp)
    dist_j = math_op("ADD", dist, math_op("ADD", warp, teeth))

    # Keep the bite clear of the canvas border no matter how the jag lands.
    border = ramp(
        math_op(
            "MINIMUM",
            math_op("MINIMUM", u, math_op("SUBTRACT", 1.0, u)),
            math_op("MINIMUM", v, math_op("SUBTRACT", 1.0, v)),
        ),
        0.015,
        0.055,
        0.0,
        1.0,
    )
    chips = math_op(
        "MULTIPLY",
        math_op("MULTIPLY", ramp(dist_j, 0.92, 1.02, 1.0, 0.0), keep),
        border,
        clamp=True,
    )
    # Fresh fracture catches light along the break line, not across the floor.
    rim = math_op("MULTIPLY", ramp(dist_j, 0.78, 0.99, 0.0, 1.0), chips, clamp=True)
    # A chip is a hole: without occlusion toward its middle the bite reads as a
    # sticker lying on the tile rather than material missing from it.
    cavity = math_op("MULTIPLY", ramp(dist_j, 0.10, 0.98, 1.0, 0.0), chips, clamp=True)

    grain = noise(uv_m, grain_cyc / max(chip_m, 1e-4), 10.0, 0.78)
    body = mix_rgb(grain, lo, hi)
    body = mix_rgb(math_op("MULTIPLY", cavity, ao_strength), body, shadow_col)
    body = mix_rgb(math_op("MULTIPLY", rim, rim_strength), body, rim_col)
    body = mix_rgb(
        math_op("MULTIPLY", math_op("MULTIPLY", joint_prox, chips), shadow_strength),
        body,
        shadow_col,
    )

    bump = node("ShaderNodeBump")
    bump.invert = True
    bump.inputs["Strength"].default_value = bump_strength
    bump.inputs["Distance"].default_value = bump_dist
    links.new(
        math_op("ADD", chips, math_op("MULTIPLY", grain, 0.25)), bump.inputs["Height"]
    )

    bsdf = node("ShaderNodeBsdfPrincipled")
    links.new(body, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    # A fresh fracture is dull biscuit or torn fibre, not glaze. Left at the
    # Principled default the specular lobe adds ~0.17 of luminance no matter
    # how dark the body is, which floats every chip off its host and makes the
    # albedo above nearly irrelevant. It is also the real cue: matte bite
    # against a glossy tile.
    for _name in ("Specular IOR Level", "Specular"):
        if _name in bsdf.inputs:
            bsdf.inputs[_name].default_value = spec
            break
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(chips, bsdf.inputs["Alpha"])

    trans = node("ShaderNodeBsdfTransparent")
    mix_sh = node("ShaderNodeMixShader")
    links.new(chips, mix_sh.inputs["Fac"])
    links.new(trans.outputs["BSDF"], mix_sh.inputs[1])
    links.new(bsdf.outputs["BSDF"], mix_sh.inputs[2])

    out = node("ShaderNodeOutputMaterial")
    links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])

    # Bump only: true displacement would sink the visible bite behind the
    # opaque host face, which is exactly where it cannot be seen.
    mat.displacement_method = "BUMP"
    mat.use_backface_culling = True
    if hasattr(mat, "blend_method"):
        mat.blend_method = "CLIP"
    if hasattr(mat, "alpha_threshold"):
        mat.alpha_threshold = 0.12
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"
    if hasattr(mat, "use_transparent_shadow"):
        mat.use_transparent_shadow = False
    return mat


def _grid_object(name: str, size: float, cuts: int) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_grid(
        bm,
        x_segments=cuts,
        y_segments=cuts,
        size=size * 0.5,
        matrix=Matrix.Identity(4),
    )
    bm.to_mesh(mesh)
    bm.free()
    return bpy.data.objects.new(name, mesh)


def _spawn_patch(site: dict, seed: int, name: str, surface: str):
    n_world = Vector(site["n_world"]).normalized()
    along = Vector(site["along"])
    if along.length < 1e-6:
        along, _ = _inplane_basis(n_world)
    else:
        along.normalize()
    across = n_world.cross(along)
    if across.length < 1e-6:
        along, across = _inplane_basis(n_world)
    else:
        across.normalize()
        along = across.cross(n_world).normalized()

    # Laminate loses whole flakes of wear layer, ceramic loses a few mm of glaze.
    chip_max = CHIP_MAX_WOOD_M if site["kind"] == "wood" else CHIP_MAX_M
    with FixedSeed(seed):
        chip_m = float(
            np.exp(np.random.uniform(math.log(CHIP_MIN_M), math.log(chip_max)))
        )
        if site["corner"]:
            chip_m = min(chip_max, chip_m * float(np.random.uniform(1.1, 1.5)))
    patch_m = float(np.clip(chip_m * PATCH_RATIO, PATCH_MIN_M, PATCH_MAX_M))

    obj = _grid_object(name, patch_m, PATCH_CUTS)
    # Local X along the joint, local Y across it into the tile, local Z out of
    # the wall - so Generated u/v line up with the shader without a swap.
    rot = Matrix((along, across, n_world)).transposed().to_4x4()
    obj.matrix_world = Matrix.Translation(site["p_world"] + n_world * PROUD_M) @ rot

    mat = create_tile_chip_material(
        f"{MATERIAL_PREFIX}_{id(obj)}",
        seed,
        kind=site["kind"],
        corner=site["corner"],
        patch_m=patch_m,
        chip_m=chip_m,
        grout_m=site["grout_m"],
        host_rgb=site.get("host_rgb"),
    )
    obj.data.materials.append(mat)
    obj["syndefect_surface"] = surface
    obj["syndefect_kind"] = "tile_chip"
    # Local Z is the outward normal for both wall and floor chips here, which
    # is not the axis pose_defect_cameras assumes for a wall; say so outright.
    obj["syndefect_normal"] = tuple(float(c) for c in n_world)
    _disable_chip_lighting(obj)
    return obj


def _iter_face_samples(obj, rng, n_try, floor: bool):
    mw = obj.matrix_world
    im3 = mw.to_3x3()
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        faces, weights = [], []
        for f in bm.faces:
            n_world = (im3 @ f.normal).normalized()
            if n_world.length < 1e-6:
                continue
            # Winding is unreliable, so both tests are on |z|; which way the
            # face actually faces is settled later by _open_side.
            if floor:
                if abs(n_world.z) < 0.72:
                    continue
            elif abs(n_world.z) > 0.38:
                continue
            area = f.calc_area()
            if area < 0.05:
                continue
            faces.append(f)
            weights.append(area)
        if not faces:
            return
        w = np.array(weights, dtype=float)
        w /= w.sum()
        for i in rng.choice(len(faces), size=n_try, replace=True, p=w):
            f = faces[i]
            verts = [v.co for v in f.verts]
            if len(verts) < 3:
                continue
            # Fan-triangulate: sampling only verts[0:3] covers half a quad, so
            # chips could never land on the far side of a wall or floor slab.
            tris = [(verts[0], verts[k], verts[k + 1]) for k in range(1, len(verts) - 1)]
            tw = np.array(
                [(t[1] - t[0]).cross(t[2] - t[0]).length for t in tris], dtype=float
            )
            if tw.sum() <= 0:
                continue
            v0, v1, v2 = tris[int(rng.choice(len(tris), p=tw / tw.sum()))]
            a, b = rng.random(), rng.random()
            if a + b > 1.0:
                a, b = 1.0 - a, 1.0 - b
            p_obj = v0 + (v1 - v0) * a + (v2 - v0) * b
            n_obj = f.normal.copy()
            if n_obj.length < 1e-6:
                continue
            n_obj.normalize()
            yield p_obj, n_obj
    finally:
        bm.free()


def _collect_sites(obj, layout, host_rgb, depsgraph, rng, n_want, floor: bool):
    kind = str(layout.get("kind") or "ceramic")
    grout_m = _mortar_m(layout)
    mw = obj.matrix_world
    m3 = mw.to_3x3()
    sites, seen = [], []
    n_off_face = n_blocked = 0

    for p_obj, n_obj in _iter_face_samples(obj, rng, n_try=640, floor=floor):
        snapped = _snap_to_joint(p_obj, n_obj, layout, max_step=0.12)
        if snapped is None:
            continue
        p_snap, du_m, dv_m, vert_grout = snapped

        seated = _reproject(obj, p_snap, n_obj)
        if seated is None:
            n_off_face += 1
            continue
        p_snap, n_obj = seated

        p_world = mw @ p_snap
        if any((p_world - q).length < 0.12 for q in seen):
            continue

        n_world = _open_side(depsgraph, p_world, (m3 @ n_obj).normalized())
        if n_world is None:
            n_blocked += 1
            continue

        # A corner site needs the perpendicular joint within a chip's reach.
        corner = max(du_m, dv_m) < CHIP_MAX_M * 0.8

        _f, grad, _du, _dv, _v = _joint_field(p_snap, n_obj, layout)
        across = (m3 @ grad).normalized() if grad.length > 1e-6 else None
        if across is None:
            continue
        along = across.cross(n_world)
        if along.length < 1e-6:
            continue
        # Gradient points away from the joint, so +across is the tile the chip
        # eats into; flipping it just picks the neighbouring tile instead.
        if rng.random() < 0.5:
            across = -across
            along = -along

        sites.append(
            {
                "p_world": p_world,
                "n_world": n_world,
                "along": along.normalized(),
                "corner": bool(corner),
                "kind": kind,
                "grout_m": grout_m,
                "host_rgb": host_rgb,
                "vert_grout": bool(vert_grout),
            }
        )
        seen.append(p_world)
        if len(sites) >= n_want * 3:
            break

    if n_off_face or n_blocked:
        logger.debug(
            "tile chips on %s: %s snapped off-face, %s with no room clearance",
            obj.name,
            n_off_face,
            n_blocked,
        )
    if not sites:
        logger.info(
            "Tile chips: no sites on %s (%s off-face, %s no clearance)",
            obj.name,
            n_off_face,
            n_blocked,
        )
        return []
    rng.shuffle(sites)
    # A mix of corner bites and edge nibbles, as in the reference photos.
    corners = [s for s in sites if s["corner"]]
    edges = [s for s in sites if not s["corner"]]
    n_corner = min(len(corners), max(1, int(round(n_want * 0.35)))) if corners else 0
    picked = corners[:n_corner] + edges[: max(0, n_want - n_corner)]
    if len(picked) < n_want:
        picked.extend(corners[n_corner : n_corner + n_want - len(picked)])
    return picked[:n_want]


def _is_floor_obj(obj) -> bool:
    n = (obj.name or "").lower()
    return n.endswith(".floor") or ".floor" in n


def _is_wall_obj(obj) -> bool:
    n = (obj.name or "").lower()
    return n.endswith(".wall") or ".wall" in n


@gin.configurable
def install_tile_chips(
    walls,
    floors=None,
    keep_rooms=None,
    enabled=True,
    n_wall=(2, 6),
    n_floor=(3, 8),
    seed=None,
):
    """Seat grout/plank-edge chips on tiled walls and jointed floors."""
    if not enabled:
        logger.info("Skipping tile chips")
        return []
    if seed is None:
        seed = _generation_seed()

    hosts = []
    for obj in list(walls or []) + list(floors or []):
        if obj is None or obj.type != "MESH":
            continue
        if not _room_is_kept(obj.name, keep_rooms):
            continue
        found = layout_from_object(obj)
        if not found:
            continue
        mat, layout = found
        # A colour the shader recorded outright beats estimating it back out
        # of the node tree.
        host_rgb = layout.get("base_color") or _mean_material_rgb(mat)
        if _is_wall_obj(obj):
            hosts.append((obj, layout, host_rgb, "wall"))
        elif _is_floor_obj(obj):
            hosts.append((obj, layout, host_rgb, "floor"))

    if not hosts:
        logger.info("Tile chips: no jointed tiled hosts")
        return []

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    col = butil.get_collection("unique_assets:tile_chips")
    spawned = []
    with FixedSeed(int_hash((seed, "tile_chips"))):
        rng = np.random.default_rng(int_hash((seed, "tile_chips_rng")))
        for i, (host, layout, host_rgb, surface) in enumerate(hosts):
            lo, hi = n_wall if surface == "wall" else n_floor
            n_want = int(rng.integers(int(lo), int(hi) + 1))
            sites = _collect_sites(
                host, layout, host_rgb, depsgraph, rng, n_want, floor=(surface == "floor")
            )
            for j, site in enumerate(sites):
                obj = _spawn_patch(
                    site,
                    int_hash((seed, "tile_chip", i, j, host.name)),
                    f"TileChip_{i:02d}_{j:02d}",
                    surface,
                )
                if obj is None:
                    continue
                butil.put_in_collection(obj, col)
                spawned.append(obj)

    if spawned:
        bpy.context.view_layer.update()
    logger.info("Tile chips: %s patches on %s tiled hosts", len(spawned), len(hosts))
    return spawned
