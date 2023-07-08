"""Script for running Simple Inpainting pipeline"""

import PIL.Image
import torch

from pipelines import StableDiffusionImg2ImgPipeline  # type: ignore

if __name__ == "__main__":
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        "redstonehero/Yiffymix_Diffusers",
        torch_dtype=torch.float16,
    )
    pipeline = pipeline.to("cuda:1")

    image = PIL.Image.open("images/image.png").convert("RGB")
    mask = PIL.Image.open("images/mask.png").convert("RGB")

    prompt = "Cute cat"
    image_result = pipeline(prompt=prompt, image=image, mask_image=mask).images[0]

    # Concat original image, mask, and inpainted image
    concat = PIL.Image.new("RGB", (512 * 3, 512))
    concat.paste(image, (0, 0))
    concat.paste(mask, (512, 0))
    concat.paste(image_result, (512 * 2, 0))

    concat.save("images/inpainting.png")
