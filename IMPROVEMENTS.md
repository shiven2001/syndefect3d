# SynDefect3D Realism Improvements

Actionable checklist to make synthetic renders look closer to real phone photos of indoor wall defects (Roboflow `indoor_defects` dataset).

**Pipeline:** Blender Cycles + Infinigen Indoors  
**Repo root:** `/home/shiven/syndefect3d/`  
**Existing dataset:** `/mnt/nvme_storage/syndefect3d_dataset_v2/`  
**Real reference stats:** mean brightness ~165, saturation ~19, close-up framing

---

## Summary: what makes current renders look “CG”

| Issue | Current behavior | Real photos |
|-------|------------------|-------------|
| Framing | Wide full-room shots (15 mm focal) | Tight close-ups of defects |
| Color | Saturated walls/tiles (sat ~85) | Flat neutrals (sat ~19) |
| Post-processing | Compositor **Contrast = 4.0** | Low contrast, flat walls |
| Lighting | Nishita sky + 1 sun lamp | Mixed ceiling + window + uneven bounce |
| Defects | Bump-mapped, “painted on” | Organic peel/curl, depth, micro-shadows |
| Sensor | Clean Cycles + denoise | Noise, soft edges, slight blur |

Model evidence of domain gap:
- Syn-only YOLO on real test: **mAP50 ≈ 0.03**
- Mixed syn+real training: **mAP50 ≈ 0.49** (vs real-only **0.15**)

---

## Recommended approach

1. Create a new Gin config **`realism_v2.gin`** (do not overwrite defaults until validated).
2. Re-render **5–10 test scenes** from existing `scene.blend` files.
3. Compare to real photos (grid + brightness/saturation stats).
4. Re-export annotations and re-run mixed-training eval.
5. Scale to full dataset once satisfied.

---

## Priority 1 — Highest impact

### 1.1 Reduce compositor contrast (fixes “CG punch”)

**File:** `infinigen/infinigen/core/rendering/render.py`  
**Function:** `compositor_postprocessing()`

**Current:**
```python
input_kwargs={"Image": source, "Bright": 1.0, "Contrast": 4.0}
```

**Change to:**
```python
input_kwargs={"Image": source, "Bright": 1.0, "Contrast": 1.2}
```

**Better (Gin-configurable):** add `@gin.configurable` parameters:
```python
@gin.configurable
def compositor_postprocessing(..., contrast=1.2, bright=1.0, distort=0.02, glare=False):
```

**Suggested values for realism v2:**
| Parameter | Default | Realism v2 |
|-----------|---------|------------|
| `Contrast` | 4.0 | **1.0 – 1.3** |
| `Bright` | 1.0 | **0.95 – 1.05** |
| `distort` | 0 | **0.01 – 0.03** (lens distortion) |
| `glare` | False | **True** (optional, for window blow-out) |

---

### 1.2 Enable defect close-up cameras (fixes framing gap)

Real dataset is macro wall shots; synthetic is mostly wide room views.

**File:** `infinigen/infinigen_examples/configs_indoor/base_indoors.gin`

**Current:**
```gin
camera.add_defect_focus.enabled = False
camera.camera_pose_proposal.focal_length = 15
```

**Change in `realism_v2.gin` (or coarse override):**
```gin
camera.add_defect_focus.enabled = True
```

> **Important:** `add_defect_focus` must be set at **coarse** generation time (`--task coarse`), not only at render. It adds one extra camera rig per defect with head-on views.

**File:** `infinigen/infinigen/core/placement/camera.py` — `pose_defect_cameras()`

**Current defaults:**
```python
distance=1.4, focal_length=40, height_jitter=0.0, horizontal_jitter=0.0, angle_noise_deg=0.0
```

**Suggested realism v2:**
```python
distance=0.9          # closer to wall
focal_length=28       # phone-like FOV (was 40 = telephoto)
height_jitter=0.08
horizontal_jitter=0.12
angle_noise_deg=6     # less perfectly head-on
```

**Coarse command example:**
```bash
python -m infinigen_examples.generate_indoors \
  --seed 51 \
  --task coarse \
  --output_folder outputs/dataset/bedroom01_v2 \
  -g bedroom_minimal.gin realism_v2.gin \
  --overrides \
    camera.spawn_camera_rigs.n_camera_rigs=10 \
    camera.add_defect_focus.enabled=True
```

---

### 1.3 Improve indoor lighting (fixes “game-engine” look)

**File:** `infinigen/infinigen_examples/configs_indoor/base_indoors.gin`  
**Override in:** `realism_v2.gin`

**Current:**
```gin
configure_render_cycles.exposure = 3
nishita_lighting.strength = 0.25
nishita_lighting.sun_elevation = ("clip_gaussian", 40, 25, 6, 70)
add_multi_directional_sun_lighting.n_directions = 1
add_multi_directional_sun_lighting.elevation_deg = 40.0
add_multi_directional_sun_lighting.energy_per_sun = 2.0
compose_indoors.lights_off_chance = 0.2
```

**Suggested realism v2:**
```gin
# Softer overall exposure (real photos are brighter but less HDR-crunchy)
configure_render_cycles.exposure = 2.0

# Weaker sky, lower sun = longer indoor shadows
nishita_lighting.strength = 0.15
nishita_lighting.sun_elevation = ("clip_gaussian", 25, 12, 10, 45)

# Multiple window directions → uneven fill light
add_multi_directional_sun_lighting.n_directions = 3
add_multi_directional_sun_lighting.elevation_deg = 30.0
add_multi_directional_sun_lighting.energy_per_sun = 1.0

# Always keep ceiling lights on (already 0.0 in bedroom_minimal.gin)
compose_indoors.lights_off_chance = 0.0
```

**Optional — phone-flash / inspection light (close-ups):**

**File:** `infinigen/infinigen/assets/lighting/sky_lighting.py` — `add_camera_based_lighting()`

Call during render or compose for defect rigs:
```python
add_camera_based_lighting(energy=("log_uniform", 80, 200), spot_size=("uniform", np.pi/5, np.pi/3))
```
Lower energy than default (200–500) to avoid harsh flash; tune per scene.

---

## Priority 2 — Color & materials

### 2.1 Desaturate wall materials

Real walls: **saturation ~19**. Synthetic: **~85**. Target: match real neutral palette.

**Where:** material resample + room surface shaders  
**File:** `infinigen/infinigen/core/rendering/resample.py` (re-seeds textures per `rs`)

**Actions:**
- Bias wall/floor colors toward beige, gray, off-white (reduce hue variance)
- Lower saturation on `shader_*` room materials in Infinigen asset definitions
- Reduce bright green/purple tile frequency in room generation (or post-filter scenes)

**Quick test:** histogram-match synthetic PNGs to real photos (see §5.3).

---

### 2.2 True geometric displacement on defects

**File:** `infinigen/infinigen_examples/configs_indoor/defect_render.gin`

**Current:**
```gin
set_displacement_mode.displacement_mode = "BUMP"
```

**Change to:**
```gin
set_displacement_mode.displacement_mode = "BOTH"
# or "DISPLACEMENT" for full geometric cracks (slower)
```

Also consider: `real_geometry_with_bump.gin` or `real_geometry.gin` for room mesh detail.

**Trade-off:** slower renders, more VRAM; much better crack/peel depth and micro-shadows.

---

### 2.3 Surface imperfection on non-defect walls

Add subtle dirt, stains, or wear shaders to plain walls so they are not perfectly clean CG surfaces.

**Where:** Infinigen material factories / `compose_indoors` material assignment  
**Reference constraint already in minimal configs:** `weak_leak_stain` in `restrict_solving.consgraph_filters`

---

## Priority 3 — Camera & sensor simulation

### 3.1 Match phone camera intrinsics

**Stored per frame:** `all_frames/<scene>/rig*_rs*/camview/camera_0/K.npz`

**Actions:**
- Set Blender sensor size + focal length to approximate iPhone (~26 mm equiv, ~73° FOV)
- Add small random perturbation to `K` per frame for diversity
- Compare focal length clusters: current wide ~853 px vs tele ~2275 px in `K`

**File:** `infinigen/infinigen/core/placement/camera.py` and coarse gin overrides

---

### 3.2 Depth of field (optional)

**File:** `infinigen/infinigen/core/rendering/render.py` — `render_image()`

**Current:** `use_dof=False`

**Suggested:**
```gin
render_image.use_dof = True
render_image.dof_aperture_fstop = 2.8   # tune: 2.0–5.6
```

Subtle blur on background edges mimics phone close-up behavior.

---

### 3.3 Film grain & noise in compositor

Add after color correction in `compositor_postprocessing()`:
- Blender compositor **Noise** node (strength ~0.02–0.05)
- Or post-process in `prepare_defect_annotated_dataset.py` export step

Real photos have **Laplacian variance ~562** vs synthetic **~220** — grain/noise helps close this gap.

---

## Priority 4 — Resample (`rs`) strategy

**What `rs0`–`rs4` means:** resample index — re-randomizes material seeds + sky lighting before each render. Camera pose is unchanged.

**File:** `infinigen/infinigen/core/rendering/resample.py`

| rs | Typical effect |
|----|----------------|
| rs0 | Original coarse materials/lighting |
| rs1–rs4 | New texture seeds + new sky lighting |

**Measured brightness (example scene):** rs0=167, rs1=147, rs2=137, **rs3=83** (very dark), rs4=174

**Recommendations:**
- Keep 5 resamples for diversity
- Optionally **exclude or cap rs3-like dark outliers** in training splits
- Or add explicit per-rs exposure override to keep brightness in real range (140–180 mean gray)
- Balance `rs` counts in train/val/test splits

---

## Priority 5 — Post-render shortcuts (no re-render)

Useful for fast ablation before committing to full re-render.

### 5.1 Image post-processing script

Apply to existing PNGs in `syndefect3d_dataset_v2_annotated/images/`:

```python
# Pseudocode — desaturate + brighten + noise
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv[:,:,1] = (hsv[:,:,1] * 0.35).astype(np.uint8)   # sat × 0.35
hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.15, 0, 255)    # brighten
img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
noise = np.random.normal(0, 4, img.shape).astype(np.int16)
img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
```

### 5.2 Random defect crops

Crop 512×512 or 640×640 regions centered on defect bboxes → matches real close-up framing without re-rendering.

### 5.3 Histogram matching

Match synthetic luminance/saturation histograms to real photo distribution per channel.

---

## New config file: `realism_v2.gin`

Create: `infinigen/infinigen_examples/configs_indoor/realism_v2.gin`

```gin
# Include after base_indoors.gin or defect_render.gin
include 'infinigen/infinigen_examples/configs_indoor/base_indoors.gin'

# --- Lighting ---
configure_render_cycles.exposure = 2.0
nishita_lighting.strength = 0.15
nishita_lighting.sun_elevation = ("clip_gaussian", 25, 12, 10, 45)
add_multi_directional_sun_lighting.n_directions = 3
add_multi_directional_sun_lighting.elevation_deg = 30.0
add_multi_directional_sun_lighting.energy_per_sun = 1.0
compose_indoors.lights_off_chance = 0.0

# --- Defect cameras (coarse stage) ---
camera.add_defect_focus.enabled = True

# --- Defect camera pose (requires gin binding on pose_defect_cameras) ---
pose_defect_cameras.distance = 0.9
pose_defect_cameras.focal_length = 28
pose_defect_cameras.height_jitter = 0.08
pose_defect_cameras.horizontal_jitter = 0.12
pose_defect_cameras.angle_noise_deg = 6

# --- Render / materials ---
set_displacement_mode.displacement_mode = "BOTH"
configure_render_cycles.num_samples = 1024
configure_render_cycles.denoise = True

# --- Compositor (after making compositor_postprocessing gin-configurable) ---
compositor_postprocessing.contrast = 1.2
compositor_postprocessing.bright = 1.0
compositor_postprocessing.distort = 0.02
compositor_postprocessing.glare = False
```

---

## Code changes checklist

Implemented on branch **`realism-v2`**. Existing gin defaults are unchanged; enable via `-g realism_v2.gin`.

| # | File | Change | Stage |
|---|------|--------|-------|
| 1 | `infinigen/core/rendering/render.py` | Gin params: contrast, bright, saturation, distort, grain (defaults still 4.0 / no grain) | Render |
| 2 | `infinigen_examples/configs_indoor/realism_v2.gin` | New overlay config | Coarse + Render |
| 3 | `infinigen_examples/configs_indoor/base_indoors.gin` | Left unchanged; override via realism_v2.gin | Coarse + Render |
| 4 | `infinigen/core/placement/camera.py` | Already gin-configurable; v2 values in realism_v2.gin | Coarse |
| 5 | `infinigen_examples/configs_indoor/defect_render.gin` | Left as BUMP; v2 sets `BOTH` | Render |
| 6 | `infinigen/assets/lighting/sky_lighting.py` | `add_camera_based_lighting.enabled` hook | Render |
| 7 | `infinigen/core/rendering/resample.py` | Optional wall/floor/ceiling desaturation | Render |
| 8 | `tools/prepare_defect_annotated_dataset.py` | `--realism-postprocess` on export | Export |

---

## Re-render workflow (existing scenes)

From `infinigen/` directory. Existing blends: `/mnt/nvme_storage/syndefect3d_dataset_v2/<room>*/scene.blend`

```bash
cd /home/shiven/syndefect3d/infinigen

OUT_ROOT="/mnt/nvme_storage/syndefect3d_dataset_v2_realism_v2"
OUT_FRAMES="${OUT_ROOT}/all_frames"
mkdir -p "${OUT_FRAMES}"

# Pilot: 3 rooms only
for room in bathroom07 bedroom18 dining21; do
  scene_dir="/mnt/nvme_storage/syndefect3d_dataset_v2/${room}"
  blend="${scene_dir}/scene.blend"
  [[ -f "${blend}" ]] || continue

  n_rigs="$(blender --background --quiet "${blend}" \
    --python tools/count_camera_rigs.py 2>/dev/null | grep -E '^[0-9]+$' | head -n1)"

  for rig in $(seq 0 $((n_rigs - 1))); do
    for rs in {0..4}; do
      python -m infinigen_examples.generate_indoors \
        --seed 0 \
        --task render \
        --input_folder "${scene_dir}" \
        --output_folder "${OUT_FRAMES}/${room}/rig${rig}_rs${rs}" \
        -g defect_render.gin realism_v2.gin \
        -p render.render_image_func=@defect/render_image \
           "execute_tasks.camera_id=[${rig},0]" \
           "execute_tasks.resample_idx=${rs}"
    done
  done
done
```

**Export annotations:**
```bash
python tools/prepare_defect_annotated_dataset.py \
  -i "${OUT_FRAMES}" \
  -o /mnt/nvme_storage/syndefect3d_dataset_v2_realism_v2_annotated
```

> For **new defect-focus rigs**, re-run **coarse** with `camera.add_defect_focus.enabled=True` instead of only re-rendering old blends.

---

## Validation checklist

After each realism change, verify:

- [ ] **Visual grid:** syn vs real side-by-side (`/mnt/nvme_storage/indoor_defects.yolo26/syn_vs_real_comparison/`)
- [ ] **Brightness:** mean gray target **150–175** (real ≈ 165)
- [ ] **Saturation:** mean sat target **15–35** (real ≈ 19)
- [ ] **Framing:** defect fills ≥30% of frame in close-up rigs
- [ ] **Zero-shot transfer:** syn-only YOLO mAP50 on real test (currently ~0.03)
- [ ] **Mixed training:** YOLO26l-p2 mAP50 on real test (currently ~0.49 mixed vs ~0.15 real-only)
- [ ] **Crack-seg transfer:** pixel IoU on syndefect3d cracks (optional)

**Stats script location:** run comparison on `/mnt/nvme_storage/syndefect3d_dataset_v2_annotated/` vs `/mnt/nvme_storage/indoor_defects.yolo26/`

---

## Suggested implementation order

1. **Compositor contrast** (1 line, huge visual change) — no re-coarse needed
2. **Re-render pilot** with new lighting gin overrides — 3 scenes
3. **Defect focus cameras** — requires re-coarse for new rigs
4. **Displacement BOTH** — re-render only
5. **Material desaturation** — re-coarse or resample tuning
6. **Post-render crop + desaturate** — fast baseline ablation
7. **Full dataset re-render** once pilot metrics improve

---

## What NOT to focus on (based on error analysis)

These were tested and are **not** the main domain gap:
- Shadow regions causing FP/FN in synthetic YOLO eval
- Lighting resample index (`rs0`–`rs4`) alone explaining detection errors
- Crack-Seg failures on grout lines (texture/appearance, not shadows)

Focus instead on: **framing, saturation, contrast, defect morphology, background texture**.

---

## References

- Main pipeline readme: `readme.md` §1–3
- Gin configs: `infinigen/infinigen_examples/configs_indoor/`
- Real dataset: [Roboflow indoor_defects](https://universe.roboflow.com/cuhk-ldjub/indoor_defects-logzf)
- Domain gap analysis: `/mnt/nvme_storage/indoor_defects.yolo26/syn_vs_real_comparison/syn_vs_real_grid.png`
- Mixed training results: `/mnt/nvme_storage/indoor_defects.yolo26/compare_experiments_l.py`

---

## Contact / notes

- Re-coarse is required for new camera rigs (`add_defect_focus`); re-render alone only changes lighting/materials per `rs`.
- Full re-render of 250 scenes × 50 rigs × 5 rs is expensive — always pilot on 3–10 scenes first.
- Document changes as **SynDefect3D v2** in thesis with before/after metrics.

----

Treat realism as done when all of this is true:

Rooms look like handover interiors (walls, ceiling, floor, openings, basic fixtures).
Lighting includes shadows at ceiling–wall junctions, not only even studio light.
Defects you care about are in the label set (crack, peel/blister, run, patchiness, stain).
You have a held-out test split that you will not train on again.