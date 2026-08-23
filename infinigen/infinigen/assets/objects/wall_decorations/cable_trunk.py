# Procedural PVC cable trunking / wire cover for walls and ceilings.

import logging
import math
from collections import defaultdict

import bpy
import numpy as np
from mathutils import Vector
from numpy.random import uniform

from infinigen.assets.objects.wall_decorations.primitives import (
    assign,
    box,
    shade_smooth,
    solid_material,
)
from infinigen.assets.utils.object import join_objects, new_bbox
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


def _pvc_color(seed_tone=None):
    tone = seed_tone if seed_tone is not None else uniform(0.78, 0.92)
    return (
        tone,
        tone * uniform(0.97, 1.0),
        tone * uniform(0.90, 0.98),
    )


class WallCableTrunkFactory(AssetFactory):
    """Horizontal PVC trunk on the wall, usually near the ceiling, optional elbow."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.length = uniform(1.15, 2.35)
            self.height = uniform(0.055, 0.085)
            self.depth = uniform(0.038, 0.055)
            self.has_elbow = uniform() < 0.7
            self.elbow_side = 1.0 if uniform() < 0.5 else -1.0
            self.elbow_len = uniform(0.12, 0.28)
            self.has_riser = uniform() < 0.55
            self.riser_h = uniform(0.08, 0.18)
            tone = uniform(0.80, 0.93)
            self.body_color = _pvc_color(tone)
            self.lid_color = _pvc_color(min(0.98, tone + uniform(0.02, 0.06)))

    def _y_extent(self):
        extra = self.elbow_len if self.has_elbow else 0.0
        lo = -self.length / 2
        hi = self.length / 2
        if self.has_elbow:
            if self.elbow_side > 0:
                hi += extra
            else:
                lo -= extra
        return lo, hi

    def create_placeholder(self, **params):
        y0, y1 = self._y_extent()
        z0 = -self.height / 2
        z1 = self.height / 2
        if self.has_riser:
            z1 = max(z1, self.height / 2 + self.riser_h)
        return new_bbox(-0.002, self.depth + 0.004, y0, y1, z0, z1)

    def create_asset(self, **params):
        body_mat = solid_material(
            f"CableTrunkBody_{self.factory_seed}",
            self.body_color,
            roughness=uniform(0.32, 0.48),
        )
        lid_mat = solid_material(
            f"CableTrunkLid_{self.factory_seed}",
            self.lid_color,
            roughness=uniform(0.28, 0.42),
        )

        run = box(
            (self.depth, self.length, self.height),
            location=(self.depth / 2, 0.0, 0.0),
            name="trunk_run",
        )
        assign(run, body_mat)
        butil.modify_mesh(run, "BEVEL", width=0.003, segments=2)
        shade_smooth(run)
        parts = [run]

        lid = box(
            (0.004, self.length * 0.97, self.height * 0.72),
            location=(self.depth + 0.001, 0.0, 0.0),
            name="trunk_lid",
        )
        assign(lid, lid_mat)
        parts.append(lid)

        clip_w = 0.018
        for t in (-0.38, 0.0, 0.38):
            clip = box(
                (0.006, clip_w, self.height * 1.08),
                location=(self.depth * 0.35, t * self.length, 0.0),
                name="trunk_clip",
            )
            assign(clip, lid_mat)
            parts.append(clip)

        if self.has_elbow:
            y_face = self.elbow_side * (self.length / 2)
            y_c = y_face + self.elbow_side * (self.elbow_len / 2)
            elbow = box(
                (self.depth, self.elbow_len, self.height),
                location=(self.depth / 2, y_c, 0.0),
                name="trunk_elbow",
            )
            assign(elbow, body_mat)
            butil.modify_mesh(elbow, "BEVEL", width=0.003, segments=2)
            parts.append(elbow)

        if self.has_riser:
            y_r = self.elbow_side * (self.length / 2)
            if self.has_elbow:
                y_r = self.elbow_side * (self.length / 2 + self.elbow_len - self.depth * 0.4)
            riser = box(
                (self.depth * 0.95, self.depth * 1.15, self.riser_h),
                location=(
                    self.depth / 2,
                    y_r,
                    self.height / 2 + self.riser_h / 2,
                ),
                name="trunk_riser",
            )
            assign(riser, body_mat)
            parts.append(riser)

        obj = join_objects(parts)
        obj.name = f"WallCableTrunkFactory({self.factory_seed}).trunk"
        return obj


class CeilingCableTrunkFactory(AssetFactory):
    """PVC trunk hung on the ceiling, typically near a wall."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.length = uniform(1.0, 2.2)
            self.width = uniform(0.050, 0.075)
            self.thick = uniform(0.032, 0.048)
            self.has_junction = uniform() < 0.45
            tone = uniform(0.80, 0.93)
            self.body_color = _pvc_color(tone)
            self.lid_color = _pvc_color(min(0.98, tone + uniform(0.02, 0.05)))

    def create_placeholder(self, **params):
        extra = self.width if self.has_junction else 0.0
        return new_bbox(
            -self.length / 2,
            self.length / 2,
            -self.width / 2,
            self.width / 2 + extra,
            -self.thick - 0.004,
            0.002,
        )

    def create_asset(self, **params):
        body_mat = solid_material(
            f"CeilTrunkBody_{self.factory_seed}",
            self.body_color,
            roughness=uniform(0.32, 0.48),
        )
        lid_mat = solid_material(
            f"CeilTrunkLid_{self.factory_seed}",
            self.lid_color,
            roughness=0.36,
        )

        run = box(
            (self.length, self.width, self.thick),
            location=(0.0, 0.0, -self.thick / 2),
            name="ceil_trunk_run",
        )
        assign(run, body_mat)
        butil.modify_mesh(run, "BEVEL", width=0.0025, segments=2)
        shade_smooth(run)
        parts = [run]

        lid = box(
            (self.length * 0.97, self.width * 0.72, 0.004),
            location=(0.0, 0.0, -self.thick - 0.001),
            name="ceil_trunk_lid",
        )
        assign(lid, lid_mat)
        parts.append(lid)

        if self.has_junction:
            junc = box(
                (self.width * 1.15, self.width * 1.15, self.thick),
                location=(
                    self.length * 0.35,
                    self.width * 0.55,
                    -self.thick / 2,
                ),
                name="ceil_trunk_junc",
            )
            assign(junc, body_mat)
            parts.append(junc)

        obj = join_objects(parts)
        obj.name = f"CeilingCableTrunkFactory({self.factory_seed}).trunk"
        return obj


logger = logging.getLogger(__name__)


def _make_trunk_run(length, width, thick, seed, ceiling=False):
    """Axis-aligned PVC run. Wall: +Y length, +X into room. Ceiling: +X length, −Z hang."""
    rng = np.random.default_rng(seed)
    tone = float(rng.uniform(0.80, 0.92))
    body_c = _pvc_color(tone)
    lid_c = _pvc_color(min(0.98, tone + 0.04))
    body_mat = solid_material(f"TrunkBody_{seed}", body_c, roughness=0.40)
    lid_mat = solid_material(f"TrunkLid_{seed}", lid_c, roughness=0.34)

    if ceiling:
        run = box((length, width, thick), location=(0.0, 0.0, -thick / 2), name="trunk")
        lid = box(
            (length * 0.98, width * 0.70, 0.004),
            location=(0.0, 0.0, -thick - 0.001),
            name="trunk_lid",
        )
    else:
        run = box((thick, length, width), location=(thick / 2, 0.0, 0.0), name="trunk")
        lid = box(
            (0.004, length * 0.98, width * 0.70),
            location=(thick + 0.001, 0.0, 0.0),
            name="trunk_lid",
        )
    assign(run, body_mat)
    butil.modify_mesh(run, "BEVEL", width=0.0025, segments=2)
    assign(lid, lid_mat)
    obj = join_objects([run, lid])
    return obj


def _place_along_xy(obj, p0, p1, z, x_axis, along_local="Y"):
    mid = (p0 + p1) * 0.5
    obj.location = (mid.x, mid.y, z)
    obj.rotation_euler = (0.0, 0.0, math.atan2(x_axis.y, x_axis.x))
    butil.apply_transform(obj, loc=False, rot=True, scale=True)
    return obj


def spawn_wall_trunk(p0, p1, inward, z, seed):
    length = (p1 - p0).length
    if length < 0.4:
        return None
    width = 0.062
    thick = 0.042
    obj = _make_trunk_run(length, width, thick, seed, ceiling=False)
    obj.name = f"WallCableTrunk_{seed}"
    _place_along_xy(obj, p0, p1, z, inward)
    return obj


def spawn_ceiling_trunk(p0, p1, z, seed):
    d = p1 - p0
    length = d.length
    if length < 0.25:
        return None
    width = 0.058
    thick = 0.038
    obj = _make_trunk_run(length, width, thick, seed, ceiling=True)
    obj.name = f"CeilingCableTrunk_{seed}"
    x_axis = Vector((d.x, d.y, 0.0))
    if x_axis.length < 1e-6:
        x_axis = Vector((1.0, 0.0, 0.0))
    else:
        x_axis.normalize()
    # Ceiling run is along local +X; rotate so +X follows the segment.
    mid = (p0 + p1) * 0.5
    obj.location = (mid.x, mid.y, z)
    obj.rotation_euler = (0.0, 0.0, math.atan2(x_axis.y, x_axis.x))
    butil.apply_transform(obj, loc=False, rot=True, scale=True)
    return obj


def spawn_junction(loc, seed, size=0.07):
    mat = solid_material(
        f"TrunkJunc_{seed}",
        _pvc_color(0.86),
        roughness=0.38,
    )
    j = box((size, size, size * 0.7), location=(loc.x, loc.y, loc.z), name="trunk_junc")
    assign(j, mat)
    j.name = f"TrunkJunction_{seed}"
    return j


def spawn_wall_riser(xy, z0, z1, inward, seed):
    """Vertical PVC on the wall, used to join an AC head to the cornice run."""
    h = abs(z1 - z0)
    if h < 0.08:
        return None
    thick = 0.042
    along = 0.072
    rng = np.random.default_rng(seed)
    tone = float(rng.uniform(0.80, 0.92))
    body_mat = solid_material(f"TrunkRiser_{seed}", _pvc_color(tone), roughness=0.40)
    run = box((thick, along, h), location=(thick / 2, 0.0, 0.0), name="trunk_riser")
    assign(run, body_mat)
    butil.modify_mesh(run, "BEVEL", width=0.0025, segments=2)
    run.name = f"WallCableRiser_{seed}"
    run.location = (xy.x, xy.y, (z0 + z1) * 0.5)
    inward = Vector((inward.x, inward.y, 0.0))
    if inward.length > 1e-6:
        inward.normalize()
        run.rotation_euler = (0.0, 0.0, math.atan2(inward.y, inward.x))
    butil.apply_transform(run, loc=False, rot=True, scale=True)
    return run


def _xy(v):
    return Vector((v.x, v.y, 0.0))


def _world_vert(obj, vi):
    return obj.matrix_world @ obj.data.vertices[vi].co


def _centroid_xy(objs):
    pts = []
    for obj in objs:
        if obj is None or obj.type != "MESH" or not obj.data.vertices:
            continue
        pts.extend(_xy(_world_vert(obj, i)) for i in range(len(obj.data.vertices)))
    if not pts:
        return None
    return sum(pts, Vector()) / len(pts)


def _boundary_edges(obj):
    counts = defaultdict(int)
    for poly in obj.data.polygons:
        verts = list(poly.vertices)
        n = len(verts)
        for i in range(n):
            e = tuple(sorted((verts[i], verts[(i + 1) % n])))
            counts[e] += 1
    return [e for e, c in counts.items() if c == 1]


def _merge_horizontal_segments(segments, centroid, min_length, z_hint=None):
    """Merge colinear axis-aligned segments into full wall/ceiling runs."""
    groups = defaultdict(lambda: [1e9, -1e9, Vector(), 0, 0.0])
    for a, b in segments:
        d = _xy(b - a)
        if d.length < 0.25:
            continue
        mid = (a + b) * 0.5
        z = a.z if z_hint is None else z_hint
        if abs(d.x) >= abs(d.y):
            key = ("x", round(mid.y, 2))
            groups[key][0] = min(groups[key][0], a.x, b.x)
            groups[key][1] = max(groups[key][1], a.x, b.x)
        else:
            key = ("y", round(mid.x, 2))
            groups[key][0] = min(groups[key][0], a.y, b.y)
            groups[key][1] = max(groups[key][1], a.y, b.y)
        inward = Vector((-d.y, d.x, 0.0))
        if inward.length > 1e-6 and centroid is not None:
            inward.normalize()
            if (centroid - _xy(mid)).dot(inward) < 0:
                inward = -inward
            groups[key][2] += inward
        groups[key][3] += 1
        groups[key][4] = z

    runs = []
    for (kind, fixed), (lo, hi, inward, n, z) in groups.items():
        if hi - lo < min_length:
            continue
        if n > 0 and inward.length > 1e-6:
            inward = (inward / n).normalized()
        else:
            inward = Vector((1.0, 0.0, 0.0))
        pad = 0.03
        if kind == "x":
            p0 = Vector((lo + pad, fixed, z))
            p1 = Vector((hi - pad, fixed, z))
        else:
            p0 = Vector((fixed, lo + pad, z))
            p1 = Vector((fixed, hi - pad, z))
        if (p1 - p0).length < min_length:
            continue
        runs.append((p0, p1, inward, z))
    return runs


def _ceiling_perimeter_runs(ceiling_obj, min_length=0.85):
    if ceiling_obj is None or ceiling_obj.type != "MESH" or not ceiling_obj.data.edges:
        return []
    segs = []
    zs = []
    edges = _boundary_edges(ceiling_obj) or [
        (e.vertices[0], e.vertices[1]) for e in ceiling_obj.data.edges
    ]
    for i0, i1 in edges:
        a = _world_vert(ceiling_obj, i0)
        b = _world_vert(ceiling_obj, i1)
        if abs(a.z - b.z) > 0.08:
            continue
        if _xy(a - b).length < 0.30:
            continue
        segs.append((a, b))
        zs.append(0.5 * (a.z + b.z))
    if not segs:
        return []
    centroid = _centroid_xy([ceiling_obj])
    return _merge_horizontal_segments(segs, centroid, min_length, z_hint=float(np.median(zs)))


def _top_cornice_runs(wall_obj, min_length=0.85):
    """Long horizontal top edges of a room wall, merged into full wall runs."""
    if wall_obj is None or wall_obj.type != "MESH" or not wall_obj.data.edges:
        return []
    zs = [_world_vert(wall_obj, i).z for i in range(len(wall_obj.data.vertices))]
    if not zs:
        return []
    z_max = max(zs)
    segs = []
    for e in wall_obj.data.edges:
        a = _world_vert(wall_obj, e.vertices[0])
        b = _world_vert(wall_obj, e.vertices[1])
        if abs(a.z - b.z) > 0.07:
            continue
        if max(a.z, b.z) < z_max - 0.12:
            continue
        if _xy(a - b).length < 0.30:
            continue
        segs.append((a, b))
    centroid = _centroid_xy([wall_obj])
    return _merge_horizontal_segments(segs, centroid, min_length, z_hint=z_max)


def _keep_interior_runs(runs, centroid):
    """Drop exterior duplicates and wall-thickness twins; keep the room-side line."""
    keep = [True] * len(runs)
    for i, (p0, p1, _in_i, _z) in enumerate(runs):
        if not keep[i]:
            continue
        mid_i = _xy((p0 + p1) * 0.5)
        dir_i = _xy(p1 - p0)
        if dir_i.length < 1e-6:
            keep[i] = False
            continue
        dir_i.normalize()
        for j in range(i + 1, len(runs)):
            if not keep[j]:
                continue
            q0, q1, _in_j, _ = runs[j]
            mid_j = _xy((q0 + q1) * 0.5)
            dir_j = _xy(q1 - q0)
            if dir_j.length < 1e-6:
                keep[j] = False
                continue
            dir_j.normalize()
            if abs(abs(dir_i.dot(dir_j)) - 1.0) > 0.18:
                continue
            sep = (mid_i - mid_j).length
            if sep > 0.45:
                continue
            if centroid is not None:
                di = (mid_i - centroid).length
                dj = (mid_j - centroid).length
                if di > dj:
                    keep[i] = False
                else:
                    keep[j] = False
            else:
                if (p1 - p0).length < (q1 - q0).length:
                    keep[i] = False
                else:
                    keep[j] = False
            if not keep[i]:
                break
    return [r for r, k in zip(runs, keep) if k]


def _named_meshes(token):
    out = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if token not in obj.name:
            continue
        if any(tok in obj.name.lower() for tok in ("cutter", "placeholder", "empty")):
            continue
        out.append(obj)
    return out


def _project_on_segment(point, a, b):
    p = _xy(point)
    a = _xy(a)
    b = _xy(b)
    ab = b - a
    if ab.length < 1e-6:
        return a, 0.0
    t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
    q = a + ab * t
    return q, (p - q).length


def _closest_on_runs(point, runs):
    best = None
    best_d = 1e9
    best_run = None
    for run in runs:
        p0, p1, inward, z = run
        q, d = _project_on_segment(point, p0, p1)
        if d < best_d:
            best_d = d
            best = Vector((q.x, q.y, z))
            best_run = run
    return best, best_d, best_run


def _in_run_bounds(pt, runs, pad=0.75):
    xs = [p.x for r in runs for p in (r[0], r[1])]
    ys = [p.y for r in runs for p in (r[0], r[1])]
    if not xs:
        return False
    return (min(xs) - pad <= pt.x <= max(xs) + pad) and (
        min(ys) - pad <= pt.y <= max(ys) + pad
    )


def _opposite_run(run, runs):
    p0, p1, inward, _z = run
    mid = _xy((p0 + p1) * 0.5)
    direc = _xy(p1 - p0)
    if direc.length < 1e-6:
        return None
    direc.normalize()
    best = None
    best_sep = 1.1
    for other in runs:
        if other is run:
            continue
        q0, q1, _oin, _ = other
        od = _xy(q1 - q0)
        if od.length < 1e-6:
            continue
        od.normalize()
        if abs(abs(direc.dot(od)) - 1.0) > 0.2:
            continue
        om = _xy((q0 + q1) * 0.5)
        sep = (om - mid).dot(inward)
        if sep > best_sep:
            best_sep = sep
            best = other
    return best


def _room_key(name):
    return name.rsplit(".", 1)[0]


def _collect_runs(walls, ceilings, min_length=0.85):
    runs = []
    srcs = [o for o in (ceilings or []) if o is not None]
    for ceil in srcs:
        try:
            runs.extend(_ceiling_perimeter_runs(ceil, min_length=min_length))
        except Exception as exc:
            logger.warning("Ceiling perimeter failed on %s: %s", ceil.name, exc)
    if not runs:
        for wall in walls or []:
            if wall is None or wall.type != "MESH":
                continue
            try:
                runs.extend(_top_cornice_runs(wall, min_length=min_length))
            except Exception as exc:
                logger.warning("Cornice extract failed on %s: %s", wall.name, exc)
    centroid = _centroid_xy(list(walls or []) + list(ceilings or []))
    return _keep_interior_runs(runs, centroid), centroid


def _put(obj, col, spawned):
    if obj is None:
        return
    butil.put_in_collection(obj, col)
    spawned.append(obj)


def install_room_cable_trunks(walls, ceilings=None):
    """Full-length wall/ceiling PVC, connected to lights and AC heads."""
    col = butil.get_collection("unique_assets:cable_trunks")
    rooms = defaultdict(lambda: {"walls": [], "ceilings": []})
    for wall in walls or []:
        rooms[_room_key(wall.name)]["walls"].append(wall)
    for ceil in ceilings or []:
        rooms[_room_key(ceil.name)]["ceilings"].append(ceil)

    lights = _named_meshes("CeilingLight")
    acs = _named_meshes("SplitAC")
    spawned = []
    seed = 1

    for key, parts in rooms.items():
        runs, _centroid = _collect_runs(parts["walls"], parts["ceilings"])
        if not runs:
            logger.info("No cornice runs for room %s", key)
            continue

        for p0, p1, inward, z in runs:
            inset = inward.normalized() * 0.014 if inward.length > 1e-6 else Vector()
            a = _xy(p0) + inset
            b = _xy(p1) + inset
            obj = spawn_wall_trunk(a, b, inward, z - 0.045, seed)
            seed += 1
            _put(obj, col, spawned)

        room_lights = [
            o for o in lights if _in_run_bounds(o.matrix_world.translation, runs)
        ]
        room_acs = [o for o in acs if _in_run_bounds(o.matrix_world.translation, runs)]
        ceil_z = max(r[3] for r in runs) - 0.010
        light_pts = [Vector((o.matrix_world.translation.x, o.matrix_world.translation.y, ceil_z)) for o in room_lights]
        spanned = set()

        for lp in light_pts:
            hit, dist, run = _closest_on_runs(lp, runs)
            if hit is None or dist > 5.0:
                continue
            opp = _opposite_run(run, runs)
            if opp is not None:
                hit2, dist2, _ = _closest_on_runs(lp, [opp])
                pair = tuple(sorted((id(run), id(opp))))
                if pair not in spanned and hit2 is not None and dist2 < 5.0:
                    obj = spawn_ceiling_trunk(_xy(hit), _xy(hit2), ceil_z, seed)
                    seed += 1
                    _put(obj, col, spawned)
                    spanned.add(pair)
                    _put(spawn_junction(Vector((hit.x, hit.y, ceil_z - 0.018)), seed, 0.055), col, spawned)
                    seed += 1
                    _put(spawn_junction(Vector((hit2.x, hit2.y, ceil_z - 0.018)), seed, 0.055), col, spawned)
                    seed += 1
                proj, off = _project_on_segment(lp, hit, hit2 if hit2 is not None else hit)
                if off > 0.14:
                    obj = spawn_ceiling_trunk(lp, Vector((proj.x, proj.y, ceil_z)), ceil_z, seed)
                    seed += 1
                    _put(obj, col, spawned)
                    _put(spawn_junction(Vector((proj.x, proj.y, ceil_z - 0.018)), seed, 0.055), col, spawned)
                    seed += 1
                elif dist > 0.12 and (hit2 is None or dist2 >= 5.0):
                    obj = spawn_ceiling_trunk(lp, _xy(hit), ceil_z, seed)
                    seed += 1
                    _put(obj, col, spawned)
                    _put(spawn_junction(Vector((hit.x, hit.y, ceil_z - 0.018)), seed, 0.055), col, spawned)
                    seed += 1
            elif dist > 0.12:
                obj = spawn_ceiling_trunk(lp, _xy(hit), ceil_z, seed)
                seed += 1
                _put(obj, col, spawned)
                _put(spawn_junction(Vector((hit.x, hit.y, ceil_z - 0.018)), seed, 0.055), col, spawned)
                seed += 1
            _put(spawn_junction(Vector((lp.x, lp.y, ceil_z - 0.020)), seed, 0.065), col, spawned)
            seed += 1

        for i, a in enumerate(light_pts):
            others = light_pts[i + 1 :]
            if not others:
                break
            b = min(others, key=lambda q: (q - a).length)
            if 0.45 < (b - a).length < 4.0:
                obj = spawn_ceiling_trunk(a, b, ceil_z, seed)
                seed += 1
                _put(obj, col, spawned)

        for ac in room_acs:
            loc = ac.matrix_world.translation
            hit, dist, run = _closest_on_runs(loc, runs)
            if hit is None or dist > 0.55 or run is None:
                continue
            zb = loc.z + 0.18
            zt = run[3] - 0.045
            if zt - zb < 0.08:
                continue
            inward = run[2]
            inset = inward.normalized() * 0.014 if inward.length > 1e-6 else Vector()
            xy = _xy(hit) + inset
            obj = spawn_wall_riser(xy, zb, zt, inward, seed)
            seed += 1
            _put(obj, col, spawned)
            _put(spawn_junction(Vector((xy.x, xy.y, zt)), seed, 0.055), col, spawned)
            seed += 1

    logger.info("Installed %s cable-trunk pieces in %s rooms", len(spawned), len(rooms))
    return spawned

