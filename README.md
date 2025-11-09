# syndefect3d


List of custom nodes used in ComfyUI:
1. ComfyUI-RTX-Remix
2. efficiency-nodes-comfyui
3. quick-connections
4. was-node-suite-comfyui
5. websocket_image_save.py
6. comfy_controlnet_preprocessors
7. comfy_mtb
8. comfyui_controlnet_aux
9. comfyui_segment_anything
10. comfyui-custom-scripts
11. ComfyUI-Easy-Use
12. ComfyUI-Manager



python -m infinigen.datagen.manage_jobs --output_folder outputs/stereo_indoors --num_scenes 30 \
--pipeline_configs local_256GB.gin stereo.gin


python -m infinigen_examples.generate_indoors --seed 0 --task coarse --output_folder outputs/indoors/coarse11 -g real_geometry.gin bedroom_minimal.gin --overrides camera.spawn_camera_rigs.n_camera_rigs=10 compute_base_views.min_candidates_ratio=20