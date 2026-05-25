## SynDefect3D

Procedurally generated photorealistic 3D synthetic dataset for indoor building defect inspection, built on [Infinigen Indoors](https://arxiv.org/abs/2406.11824). This repository extends the Infinigen pipeline with defect-focused rendering and export workflows for RGB, material segmentation, YOLO-style detection labels, and simulation use (e.g. NVIDIA Omniverse / Isaac Sim).

### Paper

_Add title, venue, and arXiv or DOI link when published._

### Citation

If you use this work, please cite our paper (see above) and the underlying Infinigen work. BibTeX for **Infinigen Indoors** (CVPR 2024):

_Add your project’s BibTeX entry here._

```bibtex
@inproceedings{infinigen2024indoors,
    author    = {Raistrick, Alexander and Mei, Lingjie and Kayan, Karhan and Yan, David and Zuo, Yiming and Han, Beining and Wen, Hongyu and Parakh, Meenal and Alexandropoulos, Stamatis and Lipson, Lahav and Ma, Zeyu and Deng, Jia},
    title     = {Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {21783-21794}
}
```

### License

This project is licensed under the **GNU General Public License v3.0** — see [`LICENSE`](LICENSE). Infinigen components under `infinigen/` follow the licenses described in [`infinigen/LICENSE`](infinigen/LICENSE).

## Dataset

**[SynDefect3D Dataset](https://huggingface.co/datasets/shiven2001/syndefect3d-dataset)** on Hugging Face — procedurally generated indoor scenes images with building defects (cracks, paint peel, spalling, bubbles, exposed wiring), built with this repository’s Infinigen-based pipeline.

**Download:** [https://huggingface.co/datasets/shiven2001/syndefect3d-dataset](https://huggingface.co/datasets/shiven2001/syndefect3d-dataset)

```python
from datasets import load_dataset
ds = load_dataset("shiven2001/syndefect3d-dataset")
# splits: train, validation, test (sample IDs per row)
```

### Splits

| Split | Rows (approx.) |
|-------|----------------|
| `train` | 19,400 |
| `validation` | 4,150 |
| `test` | 4,150 |

**Total:** 27,680 samples. Each row is a **sample id** (e.g. `bedroom18_rig15_rs4_rig15_camera_0`) encoding room type, camera rig, resample index, and camera subfolder — matching the `all_frames` layout used by `tools/prepare_defect_annotated_dataset.py`.

### Annotations

The Hub repo bundles scene assets and metadata for defect inspection. From rendered frames you can build (see §3 in this readme):

- **`images/`** — RGB frames
- **`masks/`** — semantic defect segmentation (classes 0–5; see `class_names.txt`)
- **`splits/`** — train / val / test lists

The dataset card index lists sample ids under the default subset; full imagery and masks are produced locally via the export tools or stored alongside the release assets on the Hub.

### Sample gallery

![Sample renders from SynDefect3D](sample/generated_grid.png)

*Example grid of synthetic indoor views and defect-focused wall close-ups.*

The public dataset is released under **[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html)**, consistent with this repository.

### Acknowledgements

This work builds on **[Infinigen](https://infinigen.org)** ([code](https://github.com/princeton-vl/infinigen), [documentation](https://github.com/princeton-vl/infinigen/tree/main/docs)). Please see the Infinigen repository for installation, dependencies, and the full citation list.

Additional open-source assets used in this generator:

- Air Conditioner by Daniyal Malik — [Sketchfab](https://skfb.ly/o6G6o)
- Air condition Daikin by maxsbond.work — [Sketchfab](https://skfb.ly/6R7V7)
- Indoor air conditioner unit by Rylae Shylna — [Sketchfab](https://skfb.ly/6S8W8)
- UK wall plug socket by Geng4d — [Sketchfab](https://skfb.ly/6T9X9)
- UK Plug Socket by Tenakin — [Sketchfab](https://skfb.ly/6UANY)
- Twin Plug Socket by Sousinho — [Sketchfab](https://skfb.ly/6VBOZ)
- Wall Power Outlet - Type I by cdcruz — [Sketchfab](https://skfb.ly/6WCP1)
- Grohe G-31191001 and Grohe G-32667001 by trendforward — [Sketchfab](https://skfb.ly/6XDQ2)
- Modern Faucet (high poly) by Elasta Kristya — [Sketchfab](https://skfb.ly/6YER3)
- Roca Element Bidet Mixer by Toss90 — [Sketchfab](https://skfb.ly/6ZFS4)

### Usage

Generation is driven by [Gin](https://github.com/google/gin-config) configs. See **`infinigen/infinigen_examples/configs_indoor/`** for available scenes and render settings.

Run the commands below from the **`infinigen/`** directory (or put that directory on `PYTHONPATH`), after completing Infinigen’s [installation](https://github.com/princeton-vl/infinigen/blob/main/docs/Installation.md).

**`infinigen_gpl`**, **`OcMesher`**, and required creature asset folders are **vendored in this repo** (see [`infinigen/VENDORED.md`](infinigen/VENDORED.md))—no `git submodule` step. To refresh those trees from upstream later, run `bash scripts/bootstrap_vendored_deps.sh` inside **`infinigen/`**.

#### Defect-focus cameras and YOLO / bbox exports

**Defect-focus cameras** are controlled by Gin, not by the render loop itself:

- **`camera.add_defect_focus.enabled`** (see [`infinigen_examples/configs_indoor/base_indoors.gin`](infinigen/infinigen_examples/configs_indoor/base_indoors.gin)) defaults to **`False`**. Then you only have the usual **`n_camera_rigs`** from coarse generation, all posed with the normal indoor camera logic.
- Set it to **`True`** when running **coarse** (same Gin mechanism as other overrides), e.g.  
  `-p camera.add_defect_focus.enabled=True`  
  to add **extra** camera rigs—**one additional rig per defect**—with head-on views. Your render loop should still use **`tools/count_camera_rigs.py`** so rig indices stay correct as the rig count grows.

**YOLO labels and bbox JSON** are **not** produced inside Blender during **`--task render`**. The render writes RGB + material segmentation (`MaterialSegmentation/`, `Materials/`).

- **Packaged training data** from renders: run **`tools/prepare_defect_annotated_dataset.py`** (§3). By default it writes **`images/`**, **`masks/`**, **`splits/`**, and **`class_names.txt`**.

#### 1. Coarse scene generation (example: 10 bedrooms)

```bash
for i in $(seq -w 1 10); do
  seed=$((10#$i + 50))
  out_dir="outputs/dataset/bedroom${i}"
  mkdir -p "$out_dir"
  python -m infinigen_examples.generate_indoors \
    --seed "$seed" \
    --task coarse \
    --output_folder "$out_dir" \
    -g bedroom_minimal.gin \
    --overrides camera.spawn_camera_rigs.n_camera_rigs=10
done
```

- **`--overrides camera.spawn_camera_rigs.n_camera_rigs=10`** sets how many camera rigs are stored in the blend.
- **`bedroom_minimal.gin`** (and analogous **`kitchen_minimal`**, **`dining_minimal`**, **`bathroom_minimal`**, **`livingroom_minimal`**) reduce clutter while keeping defect-related constraints; edit or combine `.gin` files as needed.

#### 2. Rendering (defect pipeline)

**`defect_render.gin`** enables material-index passes; **`execute_tasks.resample_idx`** randomizes the scene per resample. The loop below counts rigs per blend with **`tools/count_camera_rigs.py`** so you do not hard-code rig indices.

Set output roots with **`=`** (not `:`). Example:

```bash
OUT_ROOT="${OUT_ROOT:-$(pwd)/outputs/dataset}"
OUT_FRAMES_ROOT="${OUT_FRAMES_ROOT:-$(pwd)/outputs/dataset/all_frames}"
mkdir -p "${OUT_FRAMES_ROOT}"

for room in bedroom; do
  for i in $(seq -w 1 10); do
    scene_dir="${OUT_ROOT}/${room}${i}"
    [[ -d "${scene_dir}" ]] || continue
    blend="${scene_dir}/scene.blend"
    [[ -f "${blend}" ]] || continue
    n_rigs="$(blender --background --quiet "${blend}" --python tools/count_camera_rigs.py 2>/dev/null | grep -E '^[0-9]+$' | head -n1)"
    [[ "${n_rigs}" =~ ^[0-9]+$ ]] || { echo "skip ${scene_dir}: could not count camera rigs" >&2; continue; }
    for rig in $(seq 0 $((n_rigs - 1))); do
      for rs in {0..4}; do
        python -m infinigen_examples.generate_indoors \
          --seed 0 \
          --task render \
          --input_folder "${scene_dir}" \
          --output_folder "${OUT_FRAMES_ROOT}/${room}${i}/rig${rig}_rs${rs}" \
          -g infinigen_examples/configs_indoor/defect_render.gin \
          -p render.render_image_func=@defect/render_image \
             "execute_tasks.camera_id=[${rig},0]" \
             "execute_tasks.resample_idx=${rs}"
      done
    done
  done
done
```

Under each **`rig*_rs*`** folder you should see **`Image/`**, **`MaterialSegmentation/`**, and **`Materials/`** (after the compositor reorganizes outputs). Use the same layout for multiple room types by extending the `for room in ...` list (e.g. `bathroom kitchen`).

#### 3. Defect segmentation masks

**`tools/prepare_defect_annotated_dataset.py`** walks an `all_frames`-style tree and writes a flat training pack. **By default** (segmentation only):

- **`images/`**, **`masks/`** (semantic defect classes)
- **`class_names.txt`**, **`splits/`**

```bash
python tools/prepare_defect_annotated_dataset.py \
  -i /path/to/all_frames \
  -o /path/to/defect_annotated_dataset
```

Add **`--with-bboxes`** to the same command when you need loose bbox and COCO outputs from the defect planes directly. Otherwsie, we recommend building the YOLO lables from the mask segmentation instead for tight bboxes.

**One frame only** (same logic as the full exporter; add **`--with-bboxes`** for JSON/YOLO/COCO on that frame):

```bash
python tools/prepare_defect_single_sample.py \
  -i /path/to/all_frames \
  --sample-id bathroom08_rig18_rs1_rig18_camera_0 \
  -o /tmp/one_sample_export
```

#### 4. Export for Isaac Sim / Omniverse

For training embodied AI in Isaac Sim, export scenes to USD:

```bash
python -m infinigen.tools.export \
  --input_folder outputs/indoors/coarse/example \
  --output_folder outputs/my_export \
  -f usdc \
  -r 2048 \
  --omniverse
```

- **`-r 2048`** sets the resolution for exported textures.

### Contact

Questions or research discussion: **shiven@link.cuhk.edu.hk**

If this project is useful to your work, please consider citing our paper and starring the repository.
