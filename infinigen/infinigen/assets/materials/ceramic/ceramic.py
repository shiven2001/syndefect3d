# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from numpy.random import uniform

from infinigen.assets.materials.utils import common
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.random import log_uniform


def shader_ceramic(
    nw: NodeWrangler, clear=False, roughness_min=0, roughness_max=0.8, **kwargs
):
    # `clear=True` is the bathtub / sink / toilet path: glossy porcelain, not bumpy clay.
    fixture = bool(clear or kwargs.get("fixture"))
    if fixture:
        color = hsv2rgba(
            uniform(0.05, 0.14),
            uniform(0.0, 0.035),
            uniform(0.90, 0.98),
        )
        roughness_min = 0.035
        roughness_max = 0.12
    elif uniform(0, 1) < 0.8:
        color = hsv2rgba(uniform(0, 1), uniform(0.2, 0.4), log_uniform(0.3, 0.6))
    else:
        color = hsv2rgba(0, 0, log_uniform(0.3, 0.6))

    roughness = nw.build_float_curve(
        nw.musgrave(log_uniform(20, 40)), [(0, roughness_min), (1, roughness_max)]
    )
    clearcoat_roughness = nw.build_float_curve(
        nw.musgrave(log_uniform(20, 40)),
        [(0, roughness_min), (1, min(0.22, roughness_max))],
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Roughness": roughness,
            "Coat Weight": 0.22 if fixture else 1,
            "Coat Roughness": clearcoat_roughness,
            "Specular IOR Level": 0.55 if fixture else 1,
            "Base Color": color,
            "Subsurface Weight": 0.01 if fixture else uniform(0.02, 0.05),
            "Subsurface Radius": (0.02, 0.02, 0.02),
        },
    )

    out_kwargs = {"Surface": principled_bsdf}
    if not fixture:
        noise_disp = nw.new_node(
            Nodes.NoiseTexture, input_kwargs={"Scale": log_uniform(20, 40)}
        )
        musgrave_disp = nw.new_node(
            Nodes.MusgraveTexture, input_kwargs={"Scale": log_uniform(30, 60)}
        )
        disp_height = nw.scalar_add(
            nw.scalar_multiply(noise_disp.outputs["Fac"], log_uniform(0.0004, 0.0012)),
            nw.scalar_multiply(musgrave_disp, log_uniform(0.0002, 0.0006)),
        )
        displacement = nw.new_node(
            "ShaderNodeDisplacement",
            input_kwargs={
                "Height": disp_height,
                "Midlevel": 0.0000,
                "Scale": log_uniform(0.15, 0.35),
            },
        )
        out_kwargs["Displacement"] = displacement

    nw.new_node(Nodes.MaterialOutput, input_kwargs=out_kwargs)


class Ceramic:
    shader = shader_ceramic

    def generate(self):
        return surface.shaderfunc_to_material(shader_ceramic)

    @staticmethod
    def apply(obj, selection=None, clear=False, **kwargs):
        common.apply(obj, shader_ceramic, selection, clear, **kwargs)

    __call__ = generate
