# Diffusers Inpainting

![License](https://img.shields.io/github/license/Vadbeg/diffusers-inpainting)

This is a repository for image inpainting with a Stable Diffusion finetunes which
weren't trained on inpainting task. Code is based on pipeline from huggingface 🤗 Diffusers library.

It is a simple learning project, it is better to use
[StableDiffusionInpaintPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py)
or
[StableDiffusionInpaintPipelineLegacy](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint_legacy.py)
from 🤗 Diffusers library.

![inpainting](images/inpainting.png)

## Installation

1. Create a virtual environment:
```shell
virtualenv -p python3.9 .venv && source .venv/bin/activate
```
2. Install all requirements:
```shell
pip install -r requirements.txt
```
3. Use the project 🎉

## Usage

To run use command below:
```shell
python run_simple_inpainting.py --device cuda:1 --prompt "Face of a yellow cat, high resolution, sitting on a park bench" --strength 0.95 --seed 0
```

## Diffusion inpainting process

![inpainting](images/inpainting_process.gif)

This gif was created by decoding latent features at each step of the diffusion process.

## Built With

* [🤗 Diffusers](https://github.com/apple/coremltools) - Huggingface diffusion models library
* [Typer](https://typer.tiangolo.com/) - CLI framework


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

* **Vadim Titko** aka *Vadbeg* -
[LinkedIn](https://www.linkedin.com/in/vadtitko) |
[GitHub](https://github.com/Vadbeg)
