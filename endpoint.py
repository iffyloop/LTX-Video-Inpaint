import modal
from fastapi import UploadFile, Response


CUDA_VERSION = "12.4.0"
FLAVOR = "devel"
DISTRO = "ubuntu22.04"
TAG = f"{CUDA_VERSION}-{FLAVOR}-{DISTRO}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{TAG}", add_python="3.11"
).entrypoint([])

modal_image = (
    cuda_dev_image.apt_install("git", "ffmpeg", "build-essential", "clang")
    .pip_install(
        "torch==2.4.1",  # We need to install torch before installing q8_kernels, otherwise q8_kernels won't be able to find torch when it tries to compile
        "diffusers>=0.28.2",
        "transformers>=4.44.2",
        "sentencepiece>=0.1.96",
        "huggingface-hub~=0.25.2",
        "einops>=0.8.0",
        "accelerate>=1.2.1",
        "av>=14.0.1",
        "ninja>=1.11.1.3",
        "fastapi[standard]",
        "python-multipart",
    )
    .pip_install(
        "git+https://github.com/KONAKONA666/q8_kernels.git",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/root/models",
        }
    )
)
app = modal.App("ltx-video-inpaint", image=modal_image)

with modal_image.imports():
    from download_models import download_models
    from inference import init_pipeline, run_inference


@app.cls(
    gpu="A10G",  # H100
    container_idle_timeout=5 * 60,  # 5 minutes
    timeout=60 * 60,  # 1 hour
    volumes={
        "/root/models": modal.Volume.from_name(
            "ltx-video-inpaint-models", create_if_missing=True
        )
    },
)
class LTXVideoInpaintServer:
    @modal.build()
    def build(self):
        download_models()

    @modal.enter()
    def enter(self):
        self.pipeline = init_pipeline()

    @modal.web_endpoint(docs=True)
    def inference(
        self,
        init_video: UploadFile,
        latent_mask_images: list[UploadFile],
        seed: int,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        frame_rate: int,
    ):
        output_bytes = run_inference(
            init_video_filelike=init_video,
            mask_video_filelike_list=latent_mask_images,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            frame_rate=frame_rate,
        )
        return Response(content=output_bytes, media_type="video/mp4")
