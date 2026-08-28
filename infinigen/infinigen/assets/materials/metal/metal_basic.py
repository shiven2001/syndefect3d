# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from numpy.random import uniform

from infinigen.assets import colors
from infinigen.assets.materials.utils import common
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler


def shader_metal(nw: NodeWrangler, color_hsv=None, **kwargs):
    position = nw.new_node(Nodes.TextureCoord).outputs["Object"]
    noise = nw.new_node(
        Nodes.NoiseTexture, [position], input_kwargs={"Scale": uniform(10, 25)}
    )
    roughness = nw.build_float_curve(
        noise,
        [(0, uniform(0, 0.2)), (1, uniform(0.4, 0.7))],
    )
    color_hsv = color_hsv or colors.metal_hsv()
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Metallic": 1.0,
            "Specular IOR Level": uniform(0.5, 1.0),
            "Base Color": colors.hsv2rgba(color_hsv),
            "Roughness": roughness,
        },
    )
    disp_noise = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"Scale": uniform(80, 150), "Detail": uniform(6, 10)},
    )
    displacement = nw.new_node(
        Nodes.Displacement,
        input_kwargs={
            "Height": nw.scalar_multiply(disp_noise.outputs["Fac"], uniform(0.002, 0.006)),
            # Midlevel defaults to 0.5, so with a height of ~0 this pushed every
            # metal surface -0.5*Scale (150-350 mm) along its normal. Harmless as
            # a bump map, but under displacement_mode="BOTH" it is real geometry:
            # it inflated door and window furniture into blobs.
            "Midlevel": 0.0,
            "Scale": uniform(0.3, 0.7),
        },
    )
    nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf, "Displacement": displacement},
    )


class MetalBasic:
    shader = shader_metal

    def generate(self, selection=None, color_hsv=None, **kwargs):
        color_hsv = color_hsv or colors.metal_hsv()
        return surface.shaderfunc_to_material(shader_metal, color_hsv)

    def apply(self, obj, selection=None, **kwargs):
        common.apply(obj, shader_metal, selection, **kwargs)

    __call__ = generate


# The three finishes actually fitted to flat windows: black anodised, mill /
# silver aluminium, and white powder coat. `metal_hsv` roams the whole hue
# circle, which is how the frames ended up looking like random alloys.
_WINDOW_FRAME_HSV = (
    (0.00, 0.00, 0.030),  # black anodised
    (0.00, 0.00, 0.055),  # near-black, slightly open
    (0.08, 0.02, 0.520),  # mill-finish aluminium
    (0.58, 0.02, 0.620),  # cool silver
    (0.10, 0.02, 0.820),  # white powder coat
)


def shader_window_frame_metal(nw: NodeWrangler, color_hsv=None, **kwargs):
    """Flat anodised / powder-coated section.

    `shader_metal` carries a noise displacement that reads as hammered
    metalwork on a slim window section, so this keeps the tint and the gentle
    roughness break-up but leaves the surface flat.
    """
    color_hsv = color_hsv or colors.metal_hsv()
    noise = nw.new_node(
        Nodes.NoiseTexture,
        [nw.new_node(Nodes.TextureCoord).outputs["Object"]],
        input_kwargs={"Scale": uniform(60, 140)},
    )
    roughness = nw.build_float_curve(
        noise, [(0, uniform(0.24, 0.34)), (1, uniform(0.40, 0.52))]
    )
    bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": colors.hsv2rgba(color_hsv),
            # Anodised aluminium is not a mirror; part metallic reads far closer
            # than the full metallic of raw stock.
            "Metallic": uniform(0.55, 0.8),
            "Specular IOR Level": uniform(0.35, 0.5),
            "Roughness": roughness,
        },
    )
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": bsdf})


class WindowFrameMetal:
    """Anodised / powder-coated window framing."""

    shader = shader_window_frame_metal

    def __init__(self, color_hsv=None):
        if color_hsv is None:
            h, s, v = _WINDOW_FRAME_HSV[int(uniform(0, len(_WINDOW_FRAME_HSV)))]
            color_hsv = (
                h,
                min(0.10, max(0.0, s + uniform(-0.01, 0.02))),
                max(0.02, v * uniform(0.9, 1.1)),
            )
        self.color_hsv = color_hsv

    def generate(self, selection=None, color_hsv=None, **kwargs):
        return surface.shaderfunc_to_material(
            shader_window_frame_metal, color_hsv or self.color_hsv
        )

    def apply(self, obj, selection=None, **kwargs):
        kwargs.setdefault("color_hsv", self.color_hsv)
        common.apply(obj, shader_window_frame_metal, selection, **kwargs)

    __call__ = generate
