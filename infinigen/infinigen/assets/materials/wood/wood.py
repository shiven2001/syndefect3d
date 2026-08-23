# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=jDEijCwz6to by Lachlan Sarv

import numpy as np
from numpy.random import uniform

from infinigen.assets import colors
from infinigen.assets.materials.utils import common
from infinigen.assets.utils.object import new_cube
from infinigen.core import surface
from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.random import log_uniform


def get_color():
    h, s, v = colors.bark_hsv()
    return hsv2rgba(
        h + uniform(-0.0, 0.05), s + uniform(-0.3, 0.2), v * log_uniform(0.2, 20)
    )


def shader_wood(
    nw: NodeWrangler, color=None, w=None, vertical=False, floor_finish=False, **kwargs
):
    # Code generated using version 2.6.4 of the node_transpiler

    vec = nw.new_node(Nodes.TextureCoord).outputs["Object"]
    if vertical:
        vec = nw.new_node(
            Nodes.Mapping,
            [vec],
            input_kwargs={"Rotation": (np.pi / 2, 0, np.pi / 2 * np.random.randint(2))},
        )

    mapping_2 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={"Vector": vec, "Scale": (5.0000, 100.0000, 100.0000)},
    )

    if color is None:
        color = get_color()
    if w is None:
        w = uniform(0, 1)
    musgrave_texture_2 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": mapping_2,
            "W": w,
            "Scale": 10.0000,
            "Detail": 15.0000,
            "Dimension": 7.0000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    map_range_2 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": musgrave_texture_2, 3: 1.0000, 4: -1.0000},
    )

    mapping_1 = nw.new_node(Nodes.Mapping, input_kwargs={"Vector": vec})

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping_1,
            "W": w,
            "Scale": 0.5000,
            "Detail": 1.0000,
            "Distortion": 1.1000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    musgrave_texture_1 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "W": w,
            "Scale": noise_texture_1.outputs["Fac"],
            "Detail": 15.0000,
            "Dimension": 0.2000,
            "Lacunarity": 2.4000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": musgrave_texture_1, 3: -1.4000, 4: 1.5000},
    )

    map_range_1 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": map_range.outputs["Result"], 3: 1.0000, 4: 0.5000},
    )

    mapping = nw.new_node(
        Nodes.Mapping, input_kwargs={"Vector": vec, "Scale": (0.1500, 1.0000, 0.1500)}
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "W": w,
            "Detail": 5.0000,
            "Distortion": 1.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    musgrave_texture = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": noise_texture.outputs["Fac"],
            "W": w,
            "Scale": 4.0000,
            "Detail": 10.0000,
            "Dimension": 0.0000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={6: noise_texture.outputs["Fac"], 7: musgrave_texture},
        attrs={"data_type": "RGBA"},
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 0.9000, 6: map_range_1.outputs["Result"], 7: mix.outputs[2]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    mix_2 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 0.9500, 6: map_range_2.outputs["Result"], 7: mix_1.outputs[2]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    hue_saturation_value = nw.new_node(
        "ShaderNodeHueSaturation",
        input_kwargs={
            "Saturation": 0.8000,
            "Value": 0.2000,
            "Color": color,
        },
    )

    mix_3 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: mix_2.outputs[2],
            6: hue_saturation_value,
            7: color,
        },
        attrs={"data_type": "RGBA"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix_2.outputs[2], 1: log_uniform(0.0012, 0.015)},
        attrs={"operation": "MULTIPLY"},
    )

    displacement = nw.new_node(
        "ShaderNodeDisplacement",
        input_kwargs={
            "Height": multiply,
            "Midlevel": 0.0000,
            "Scale": log_uniform(0.3, 0.9),
        },
    )

    color = mix_3.outputs[2]
    if kwargs.get("floor_finish") or floor_finish:
        # Satin / matte polyurethane — not wet plastic.
        roughness = uniform(0.38, 0.52)
        roughness = nw.build_float_curve(
            nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": log_uniform(22, 40)}),
            [(0, roughness), (1, roughness + uniform(0.08, 0.20))],
        )
        coat = uniform(0.04, 0.14)
        spec = uniform(0.20, 0.36)
    else:
        roughness = uniform(0.0, 0.4)
        roughness = nw.build_float_curve(
            nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": log_uniform(40, 50)}),
            [(0, roughness), (1, roughness + uniform(0.0, 0.8))],
        )
        coat = np.clip(uniform(0, 1.4), 0, 1)
        spec = None
    principled_kwargs = {
        "Base Color": color,
        "Roughness": roughness,
        "Coat Weight": coat,
    }
    if spec is not None:
        principled_kwargs["Specular IOR Level"] = spec
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs=principled_kwargs,
    )
    nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf, "Displacement": displacement},
    )


def shader_wood_floor(nw: NodeWrangler, color=None, w=None, vertical=False, **kwargs):
    """Interior floor boards: same grain as wood, satin (not mirror) finish."""
    kwargs.pop("floor_finish", None)
    shader_wood(
        nw, color=color, w=w, vertical=vertical, floor_finish=True, **kwargs
    )


class Wood:
    shader = shader_wood

    def generate(self):
        return surface.shaderfunc_to_material(shader_wood)

    def apply(self, obj, selection=None, **kwargs):
        common.apply(obj, shader_wood, selection, **kwargs)

    __call__ = generate


# Typical interior door / window-frame woods (oak, walnut, pine, maple, mahogany).
_INTERIOR_WOOD_HSV = (
    (0.075, 0.42, 0.38),  # walnut
    (0.065, 0.48, 0.28),  # dark walnut
    (0.082, 0.38, 0.48),  # oak
    (0.090, 0.32, 0.58),  # light oak
    (0.085, 0.22, 0.64),  # pine
    (0.055, 0.50, 0.34),  # mahogany
    (0.070, 0.18, 0.72),  # maple
    (0.080, 0.28, 0.42),  # teak
)


def sample_interior_wood_color():
    h, s, v = _INTERIOR_WOOD_HSV[int(np.random.randint(len(_INTERIOR_WOOD_HSV)))]
    return hsv2rgba(
        h + uniform(-0.015, 0.015),
        float(np.clip(s + uniform(-0.08, 0.08), 0.12, 0.62)),
        float(np.clip(v + uniform(-0.08, 0.08), 0.22, 0.82)),
    )


class InteriorWood:
    """Wood shader with apartment-door color range (not random bark / plywood)."""

    shader = shader_wood

    def __init__(self, color=None, w=None, **kwargs):
        self.color = color if color is not None else sample_interior_wood_color()
        self.w = w if w is not None else uniform(0, 1)
        self.extra = kwargs

    def generate(self, **kwargs):
        color = kwargs.pop("color", self.color)
        w = kwargs.pop("w", self.w)
        return surface.shaderfunc_to_material(shader_wood, color=color, w=w, **kwargs)

    def apply(self, obj, selection=None, **kwargs):
        color = kwargs.pop("color", self.color)
        w = kwargs.pop("w", self.w)
        kwargs.pop("metal_color", None)
        common.apply(obj, shader_wood, selection, color=color, w=w, **kwargs)

    __call__ = generate


def make_sphere():
    return new_cube()
