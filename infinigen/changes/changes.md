/home/shiven/infinigen/infinigen_examples/configs_indoor/bathroom_minimal.gin
bedroom bathroom livingroom kitchen..

also code will be:



/home/shiven/infinigen/infinigen/core/tags.py

/home/shiven/infinigen/infinigen_examples/constraints/semantics.py

/home/shiven/infinigen/infinigen_examples/constraints/home.py



python -m infinigen.tools.export --input_folder {PATH_TO_FOLDER_OF_BLENDFILES} --output_folder outputs/my_export -f usdc -r 1024

If you want a different output format, please use the "--help" flag or use one of the options below:

    -f obj will export in .obj format,
    -f fbx will export in .fbx format
    -f stl will export in .stl format
    -f ply will export in .ply format.
    -f usdc will export in .usdc format.
    -v enables per-vertex colors (only compatible with .fbx and .ply formats).
    -r {INT} controls the resolution of the baked texture maps. For instance, -r 1024 will export 1024 x 1024 texture maps.
    --individual will export each object in a scene in its own individual file.
    --omniverse will prepare the scene for import to IsaacSim or other NVIDIA Omniverse programs. See more in Exporting to Simulators.

python -m infinigen.tools.export --input_folder outputs/indoors/coarse --output_folder outputs/my_export -f usdc -r 4096 --omniverse