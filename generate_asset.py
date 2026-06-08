#!/usr/bin/env python3
"""Generate a textured GLB from an image or text prompt via Hunyuan3D-2.

Usage:
    python generate_asset.py --image <image_path> <output.glb>
    python generate_asset.py --text "a red fire truck with aerial ladder, boxy body, chrome details" <output.glb>

Batch (image):
    for img in concept_sketches/*.png; do
        python generate_asset.py --image "$img" "game_assets/$(basename "${img%.png}").glb"
    done
"""
import os, sys, gc, argparse

HF_CACHE = "/apps/models/huggingface"
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = f"{HF_CACHE}/hub"
os.environ["HY3DGEN_MODELS"] = "/apps/models/hy3dgen"

import torch
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline


def generate(image, output_glb: str) -> None:
    """image can be a file path string or a PIL Image."""
    print(f"[shape] Shape generation")
    shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
    mesh = shape_pipe(image=image)[0]
    del shape_pipe
    gc.collect()
    torch.cuda.empty_cache()

    print("[tex]   Texture painting")
    tex_pipe = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    mesh = tex_pipe(mesh, image=image)
    del tex_pipe

    os.makedirs(os.path.dirname(os.path.abspath(output_glb)), exist_ok=True)
    mesh.export(output_glb)
    print(f"Done: {output_glb}")


def generate_from_text(prompt: str, output_glb: str) -> None:
    from hy3dgen.text2image import HunyuanDiTPipeline
    from hy3dgen.rembg import BackgroundRemover
    print(f"[t2i]   Text → image: {prompt!r}")
    t2i = HunyuanDiTPipeline(
        model_path="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"
    )
    image = t2i(prompt)
    del t2i
    gc.collect()
    torch.cuda.empty_cache()

    print("[rembg] Removing background")
    image = BackgroundRemover()(image)
    preview_path = output_glb.replace(".glb", "_preview.png")
    image.save(preview_path)
    print(f"[rembg] Preview saved: {preview_path}")

    generate(image, output_glb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hunyuan3D-2 asset generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", metavar="PATH", help="Input image path")
    group.add_argument("--text", metavar="PROMPT", help="Text prompt for text-to-3D")
    parser.add_argument("output_glb", metavar="output.glb")
    args = parser.parse_args()

    if args.image:
        generate(args.image, args.output_glb)
    else:
        generate_from_text(args.text, args.output_glb)
