import io
import json
import random
from pathlib import Path

import av
import numpy as np
import safetensors.torch
import torch
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer
from q8_kernels.models.LTXVideo import LTXTransformer3DModel
from q8_kernels.graph.graph import make_dynamic_graphed_callable

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.conditioning_method import ConditioningMethod

from ltx_video.models.transformers.transformer3d import Transformer3DModel

TRANSFORMER_TYPE = "q8_kernels"
LOW_VRAM = False
MODELS_DIR = Path("/root/models")
VAE_DIR = MODELS_DIR / "Lightricks--LTX-Video/vae"
UNET_PATH = MODELS_DIR / "konakona--ltxvideo_q8"
SCHEDULER_DIR = MODELS_DIR / "Lightricks--LTX-Video/scheduler"
TEXT_ENCODER_DIR = MODELS_DIR / "Lightricks--LTX-Video"
TOKENIZER_DIR = MODELS_DIR / "PixArt-alpha--PixArt-XL-2-1024-MS"
VAE_SCALE_FACTOR = (
    32  # Hardcoded here for convenience, but can also be determined programatically
)
VIDEO_SCALE_FACTOR = 8  # See above comment


def load_vae(vae_dir):
    vae_ckpt_path = vae_dir / "vae_diffusion_pytorch_model.safetensors"
    vae_config_path = vae_dir / "config.json"
    with open(vae_config_path, "r") as f:
        vae_config = json.load(f)
    vae = CausalVideoAutoencoder.from_config(vae_config)
    vae_state_dict = safetensors.torch.load_file(vae_ckpt_path)
    vae.load_state_dict(vae_state_dict)
    if torch.cuda.is_available():
        vae = vae.cuda()
    return vae.to(torch.bfloat16)


def load_unet(unet_path, type="q8_kernels"):
    if type == "q8_kernels":
        transformer = LTXTransformer3DModel.from_pretrained(unet_path)
    else:
        unet_ckpt_path = unet_path / "unet_diffusion_pytorch_model.safetensors"
        unet_config_path = unet_path / "config.json"
        transformer_config = Transformer3DModel.load_config(unet_config_path)
        transformer = Transformer3DModel.from_config(transformer_config)
        unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
        transformer.load_state_dict(unet_state_dict, strict=True)
        if torch.cuda.is_available():
            transformer = transformer.cuda()

    return transformer


def load_scheduler(scheduler_dir):
    scheduler_config_path = scheduler_dir / "scheduler_config.json"
    scheduler_config = RectifiedFlowScheduler.load_config(scheduler_config_path)
    return RectifiedFlowScheduler.from_config(scheduler_config)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_video_to_tensor(video_filelike):
    container = av.open(video_filelike)
    frames = []
    for frame in container.decode(video=0):
        frame = frame.to_ndarray(format="rgb24").astype(np.float32) / 127.5 - 1.0
        frames.append(frame)
    frames = np.array(frames)
    frames = np.permute_dims(frames, (3, 0, 1, 2))  # Permute F, H, W, C to C, F, H, W
    return torch.tensor([frames]).float()  # B, C, F, H, W


def load_video_mask_images_to_tensor(mask_filelike_list):
    images = []
    for mask_filelike in mask_filelike_list:
        image = np.array(Image.open(mask_filelike))
        image = np.mean(image, axis=2)
        image = image.reshape((image.shape[0], image.shape[1], 1))  # H, W -> H, W, C
        images.append(image.astype(np.float32) / 255)
    images = np.array(images)
    images = np.permute_dims(images, (3, 0, 1, 2))  # Permute F, H, W, C to C, F, H, W
    return torch.tensors([images]).float()  # B, C, F, H, W


def init_pipeline():
    # Load models
    vae = load_vae(VAE_DIR)
    unet = load_unet(UNET_PATH, type=TRANSFORMER_TYPE)
    scheduler = load_scheduler(SCHEDULER_DIR)
    patchifier = SymmetricPatchifier(patch_size=1)
    text_encoder = T5EncoderModel.from_pretrained(
        TEXT_ENCODER_DIR, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    if torch.cuda.is_available() and not LOW_VRAM:
        text_encoder = text_encoder.to("cuda")

    tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_DIR, subfolder="tokenizer")

    unet = unet.to(torch.bfloat16)
    if TRANSFORMER_TYPE == "q8_kernels":
        for b in unet.transformer_blocks:
            b.to(dtype=torch.float)

        for n, m in unet.transformer_blocks.named_parameters():
            if "scale_shift_table" in n:
                m.data = m.data.to(torch.bfloat16)

        # unet = unet.cuda()
        torch.cuda.synchronize()
        unet.forward = make_dynamic_graphed_callable(unet.forward)
    else:
        pass

    # Use submodels for the pipeline
    submodel_dict = {
        "transformer": unet,
        "patchifier": patchifier,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "vae": vae,
    }

    pipeline = LTXVideoPipeline(**submodel_dict)
    if torch.cuda.is_available() and not LOW_VRAM:
        pipeline = pipeline.to("cuda")

    return pipeline


def run_inference(
    pipeline: LTXVideoPipeline,
    init_video_filelike,
    mask_video_filelike_list,
    seed: int,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 20,
    guidance_scale: float = 3.0,
    frame_rate: int = 25,
):
    seed_everything(seed)

    media_items = load_video_to_tensor(init_video_filelike)
    media_items_mask = load_video_mask_images_to_tensor(mask_video_filelike_list)

    height = media_items.shape[-2]
    width = media_items.shape[-1]
    num_frames = media_items.shape[-3]

    # Prepare input for the pipeline
    sample = {
        "prompt": prompt,
        "prompt_attention_mask": None,
        "negative_prompt": negative_prompt,
        "negative_prompt_attention_mask": None,
        "media_items": media_items,
        "media_items_mask": media_items_mask,
    }

    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(seed)

    images = pipeline(
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,  # Is this correct, if we're using an input video?
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
        callback_on_step_end=None,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        **sample,
        is_video=True,
        vae_per_channel_normalize=True,
        conditioning_method=(
            ConditioningMethod.FIRST_FRAME
            if media_items is not None
            else ConditioningMethod.UNCONDITIONAL
        ),
        mixed_precision=False,
        low_vram=LOW_VRAM,
        transformer_type=TRANSFORMER_TYPE,
    ).images[0]

    out_video_np = (
        images.permute(1, 2, 3, 0).cpu().float().numpy()  # C, F, H, W -> F, H, W, C
    )
    out_video_np = (out_video_np * 255).astype(np.uint8)
    out_height, out_width = out_video_np.shape[1:3]
    out_bytesio = io.BytesIO()
    out_container = av.open(out_bytesio, "w")
    out_stream = out_container.add_stream("mpeg4", rate=frame_rate)
    out_stream.width = out_width
    out_stream.height = out_height
    out_stream.pix_fmt = "yuv420p"
    for frame_i in out_video_np.shape[0]:
        frame = av.VideoFrame.from_ndarray(out_video_np[frame_i], format="rgb24")
        for packet in out_stream.encode(frame):
            out_container.mux(packet)
    for packet in out_stream.encode():
        out_container.mux(packet)
    out_container.close()
    return out_bytesio.getbuffer()
