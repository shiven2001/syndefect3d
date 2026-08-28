# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

SynDefect3D generates a synthetic, photorealistic dataset of indoor building defects (cracks, paint
peel/blister, spalling, paint bubbles, water stains, exposed wiring, paint runs/patches) for training
defect-inspection models (YOLO detection, U-Net/segmentation). It is a fork of **Infinigen Indoors**
(procedural indoor scene generation in Blender/Cycles), extended with:

- Procedural defect assets (crack, spalling, paint peel, wall bubbles, paint run/patch planes) placed
  as wall/ceiling decorations by the constraint solver.
- Defect-focused close-up cameras, in addition to Infinigen's normal room-trajectory cameras.
- Export tooling that turns rendered frames + material-index passes into segmentation masks, YOLO boxes,
  and COCO annotations keyed by defect material name.

Repo layout:
- `infinigen/` — the vendored Infinigen fork itself (all engine code, gin configs, docs, tests). This is
  where nearly all real work happens.
- `defect_generation/` — standalone `.blend` files and one-off node-based Python generators
  (`procedural_*_plane_gen.py`) used as references/prototypes for the "real" procedural factories now
  living under `infinigen/infinigen/assets/*_plane.py`. Not part of the runtime pipeline.
- `sample/` — example output imagery for the readme.
- `IMPROVEMENTS.md` — active checklist for the realism-v2 effort (see "Current branch" below).

The public dataset built from this pipeline is on Hugging Face
(`shiven2001/syndefect3d-dataset`); `readme.md` documents its layout and the annotation/export flow in
more detail.

## Commands

All commands below assume `cd infinigen/` first (the installable package root), and a Python 3.11 env
with Blender-as-python (`bpy==4.2.0`).

**Install:**
```bash
pip install -e .                        # minimal
pip install -e ".[dev,terrain,vis]"     # full dev install (tests, ruff, terrain, visualization)
```

**Generate a scene (two-stage: coarse layout, then render):**
```bash
# 1. Coarse: solves room layout + object/defect placement, writes scene.blend
python -m infinigen_examples.generate_indoors --seed 0 --task coarse \
  --output_folder outputs/indoors/coarse -g fast_solve.gin singleroom.gin \
  -p compose_indoors.terrain_enabled=False restrict_solving.restrict_parent_rooms=\[\"Bedroom\"\]

# 2. Render: renders RGB + ground truth from an existing scene.blend
python -m infinigen_examples.generate_indoors --seed 0 --task render \
  --input_folder outputs/indoors/coarse --output_folder outputs/indoors/frames
```
`-g <name>.gin` selects gin config overlays (from `infinigen_examples/configs_indoor/`); `-p key=value`
sets individual gin bindings on the command line. Camera rig count for a given `scene.blend` can be
checked with `tools/count_camera_rigs.py` (run via `blender --background --quiet scene.blend --python
tools/count_camera_rigs.py`).

**Generate a full dataset (parallelized across scenes):**
```bash
python -m infinigen.datagen.manage_jobs --output_folder outputs/my_dataset --num_scenes 1000 \
  --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin indoor_background_configs.gin \
  --configs singleroom.gin \
  --pipeline_overrides get_cmd.driver_script='infinigen_examples.generate_indoors' manage_datagen_jobs.num_concurrent=16 \
  --overrides compose_indoors.restrict_single_supported_roomtype=True
```

**Export annotations from rendered frames** (RGB + material-index pass -> masks/bboxes/YOLO/COCO):
```bash
python tools/prepare_defect_annotated_dataset.py -i <all_frames_dir> -o <output_dir> [--with-bboxes] [--force]
python tools/prepare_defect_single_sample.py   # single-sample debug variant
python tools/visualize_defect_yolo_annotations.py
```

**Tests:**
```bash
pytest tests                                    # full suite (testpaths = tests, 480s per-test timeout)
pytest tests/assets/test_placeholders.py -k foo # single test
pytest -m indoors                               # marker-filtered (markers: nature, indoors, skip_for_ci)
```

**Lint:**
```bash
ruff check .
ruff format .
isort .
```

**Blender UI (inspect a generated `scene.blend`):**
```bash
python -m infinigen.launch_blender outputs/indoors/coarse/scene.blend
```
If Infinigen was installed as a standalone Blender-Python script rather than into a system Python env,
prefix any `python -m <module> <args>` command with `python -m infinigen.launch_blender -m <module> --`.

## Architecture

**Two-stage pipeline.** `infinigen_examples/generate_indoors.py` is the entry point for both stages,
dispatched by `--task coarse|render` (see `execute_tasks` in `infinigen/core`). Coarse solves room
geometry and object/defect placement via a constraint solver and saves a `.blend`; render loads that
`.blend`, positions cameras, and runs Cycles. Camera rig count, and therefore which defects get
close-up coverage, is fixed at coarse time — re-rendering an existing `scene.blend` cannot add new rigs.

**Constraint-based placement.** Object and defect placement is not scripted directly — it's declared as
constraints over `Semantics` tags and solved by `infinigen/core/constraints/example_solver`.
`infinigen_examples/constraints/home.py` (`home_furniture_constraints()` and friends) is where defects
are wired in: `Semantics.Defects` is its own category (separate from generic wall decorations — see the
comment block around line 660), split into `defects_wall` / `defects_ceiling` via
`related_to(rooms, cu.flush_wall_defect / flush_ceiling_defect)`, then further split per defect type
(`CrackPlaneFactory`, `PaintPeelPlaneFactory`, `WallBubblePlaneFactory`, `PaintRunPlaneFactory`,
`PaintPatchPlaneFactory`, plus ceiling variants). `infinigen_examples/constraints/semantics.py` defines
which asset factories belong to `Semantics.Defects`. Some factories (spalling, spalling-plug, weak leak
stain, open wiring) are currently commented out of the constraint graph in `home.py` — re-enable there,
not just in the factory files, to bring them back into generated scenes.
`restrict_solving.consgraph_filters` (set per-gin-config) can further prune which named constraints run,
e.g. the `*_minimal.gin` configs keep only `["cracks", "paint_peel", "wall_bubbles", "paint_runs",
"paint_patches", ...]` plus fixtures, to produce empty rooms with defects and nothing else.

**Defect assets.** Each defect type is a procedural `AssetFactory` under `infinigen/infinigen/assets/`
(e.g. `crack_plane.py`, `spalling_plane.py`, `paint_peel_plane.py`) that builds a thin plane mesh + a
Cycles shader material driven by numpy-randomized Voronoi/noise parameters, tagged for canonical-surface
placement (`tag_canonical_surfaces`). `defect_generation/procedural_*_plane_gen.py` and the `.blend`
files in `defect_generation/procedural_blender_files/` are the original hand-built node-graph prototypes
these factories were adapted from — useful as a reference when tuning shader parameters, not imported at
runtime.

**Defect-focus cameras.** Normal Infinigen indoor cameras follow room trajectories
(`infinigen/core/placement/camera.py`, `infinigen/core/placement/camera_trajectories.py`). This fork adds
`pose_defect_cameras()` (same file) and a `camera.add_defect_focus.enabled` gin flag: when enabled at
**coarse** time, one extra close-up camera rig is added per placed defect, aimed head-on at short
distance. `defect_camera_fraction()` in `generate_indoors.py` is a currently-unused alternative
(reserve a fraction of the *main* rig batch for defect posing instead of adding extra rigs; defaults to
off).

**Material -> class mapping for exports.** `tools/prepare_defect_annotated_dataset.py` reads the RGB
image, the `MaterialSegmentation` index-map pass, and the `Materials` name->pass_index JSON that
Infinigen's ground-truth extraction produces per camera, then maps material name prefixes
(`CrackMaterial_*`, `PaintPeelMaterial_*`, `SpallingMaterial_*`/`SpallingPlugMaterial_*`,
`BubbleMaterial_*`, `OpenWiringMaterial_*`, `PaintRunMaterial_*`, `PaintPatchMaterial_*`) to fixed class
IDs 1-7 (0 = background) to produce masks, and optionally per-material-pass bounding boxes in both
YOLO-normalized and COCO format. Adding a new defect type means: new `AssetFactory` + material naming
convention, wire it into `home.py`'s `Semantics.Defects` constraints, then extend this class map.

**Gin config layering.** `infinigen_examples/configs_indoor/base_indoors.gin` holds pipeline defaults;
scene-shape presets (`singleroom.gin`, `fast_solve.gin`, `overhead.gin`, ...) and this fork's
`*_minimal.gin` configs (bedroom/bathroom/kitchen/dining/livingroom/studio_apartment) layer on top via
`-g`, and `-p key=value` overrides individual bindings from the command line. `*_minimal.gin` configs are
the "empty room, defects + fixtures only" presets used for the defect dataset: they disable
small/large-object solving, keep only medium-object solving (where defect planes live), force one room
type via `restrict_solving.restrict_parent_rooms`, and prune `consgraph_filters` down to defect + fixture
constraint names.

## Current branch: `realism-v2`

Working from `IMPROVEMENTS.md`, a prioritized checklist to close the synthetic-vs-real domain gap
(measured via YOLO mAP50 on a real Roboflow `indoor_defects` test set — currently ~0.03 syn-only,
~0.49 mixed syn+real vs ~0.15 real-only). Priority order: compositor contrast (`Contrast: 4.0 -> ~1.2` in
`compositor_postprocessing()`, `infinigen/infinigen/core/rendering/render.py`), defect close-up framing
(`camera.add_defect_focus`), lighting mix, material desaturation, and defect morphology/noise. The
intended workflow is a new `realism_v2.gin` overlay (existing defaults untouched) validated on a small
pilot (3-10 scenes) against brightness/saturation stats and detection mAP before scaling to a full
re-render. Recent commit `8f1f8c1` ("Make empty-apartment scenes look like lived-in interiors") and the
many modified files under `infinigen/infinigen/assets/objects/{bathroom,appliances,...}` are part of
this effort — check `git status`/`git diff` for the current in-progress state rather than assuming
`IMPROVEMENTS.md` is fully implemented.
