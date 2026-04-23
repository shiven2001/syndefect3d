## SynDefect3D

Procedurally generated photorealistic 3D synthetic dataset for indoor building defect inspection, built on [Infinigen Indoors](https://arxiv.org/abs/2406.11824). This repository extends the Infinigen pipeline with defect-focused rendering and export workflows for RGB, material segmentation, and simulation use (e.g. NVIDIA Omniverse / Isaac Sim).

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

_Add dataset name, download link, and a short description of splits and annotations when released._

### Acknowledgements

This work builds on **[Infinigen](https://infinigen.org)** ([code](https://github.com/princeton-vl/infinigen), [documentation](https://github.com/princeton-vl/infinigen/tree/main/docs)). Please see the Infinigen repository for installation, dependencies, and full citation list.

Additional open-source assets were used in this generator. These are:

(ADD)
\begin{enumerate}[-]
    \item Air Conditioner by Daniyal Malik (\href{https://skfb.ly/o6G6o}{Source}).
    \item Air condition Daikin by maxsbond.work (\href{https://skfb.ly/6R7V7}{Source}).
    \item Indoor air conditioner unit by Rylae Shylna (\href{https://skfb.ly/6S8W8}{Source}).
    \item UK wall plug socket by Geng4d (\href{https://skfb.ly/6T9X9}{Source}).
    \item UK Plug Socket by Tenakin (\href{https://skfb.ly/6UANY}{Source}).
    \item Twin Plug Socket by Sousinho (\href{https://skfb.ly/6VBOZ}{Source}).
    \item Wall Power Outlet - Type I by cdcruz (\href{https://skfb.ly/6WCP1}{Source}).
    \item Grohe G-31191001 and Grohe G-32667001 by trendforward (\href{https://skfb.ly/6XDQ2}{Source}).
    \item Modern Faucet (high poly) by Elasta Kristya (\href{https://skfb.ly/6YER3}{Source}).
    \item Roca Element Bidet Mixer by Toss90 (\href{https://skfb.ly/6ZFS4}{Source}).
\end{enumerate}
---

### Usage

Generation is driven by [Gin](https://github.com/google/gin-config) configs. See **`infinigen/infinigen_examples/configs_indoor/`** for available scenes and render settings.

Run the commands below from the **`infinigen/`** directory (or ensure that directory is on `PYTHONPATH`), after completing Infinigen’s [installation](https://github.com/princeton-vl/infinigen/blob/main/docs/Installation.md).

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

- **`--overrides camera.spawn_camera_rigs.n_camera_rigs=10`** increases the number of camera rigs in the scene.
- **`-g bedroom_minimal.gin`** uses a minimal room: structural geometry only, without extra furniture. The paper uses five analogous configs: **`bedroom_minimal`**, **`kitchen_minimal`**, **`dining_minimal`**, **`bathroom_minimal`**, and **`livingroom_minimal`**. You can edit these, combine with other `.gin` files in that folder, or add your own.

#### 2. Rendering (defect pipeline)

Because coarse generation used 10 camera rigs, render once per rig. **`defect_render.gin`** controls the render setup; **`execute_tasks.resample_idx`** runs multiple resamples per camera for domain randomization (5 resamples in this example).

```bash
OUT_ROOT="${OUT_ROOT:-$(pwd)/outputs/dataset}"
OUT_FRAMES_ROOT="${OUT_FRAMES_ROOT:-$(pwd)/outputs/dataset/all_frames}"
mkdir -p "${OUT_FRAMES_ROOT}"

for room in bedroom; do
  for i in $(seq -w 1 10); do
    [[ -d "${OUT_ROOT}/${room}${i}" ]] || continue
    for rig in {0..9}; do
      for rs in {0..4}; do
        python -m infinigen_examples.generate_indoors \
          --seed 0 \
          --task render \
          --input_folder "${OUT_ROOT}/${room}${i}" \
          --output_folder "${OUT_FRAMES_ROOT}/${room}${i}/rig${rig}_rs${rs}" \
          -g infinigen_examples/configs_indoor/defect_render.gin \
          -p render.render_image_func=@defect/render_image \
             execute_tasks.camera_id=[${rig},0] \
             execute_tasks.resample_idx=${rs}
      done
    done
  done
done
```

This produces RGB frames and full material segmentation. To convert them into RGB plus **pixel masks for defects only**, use the post-processing script _(to be added)_.

#### 3. Export for Isaac Sim / Omniverse

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

---

### Contact

Questions or research discussion: **shiven@link.cuhk.edu.hk**

If this project is useful to your work, please consider citing our paper and starring the repository.
