import random

import numpy as np
import torch
import imageio

from LTXVideoSession import LTXVideoSession


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


if __name__ == "__main__":
    ckpt_path = "checkpoints/ltx-video-2b-v0.9.5.safetensors"
    prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
    negative_prompt = "worst quality, inconsistent motion, blurry, dof, depth of field, motion blur, jittery, distorted, jpeg artifacts"
    enhance_prompt = False
    seed = 3
    num_inference_steps = 25
    width = 704
    height = 480
    num_frames = 25
    frame_rate = 25

    conditioning_videos = ["inputs/2/tokyo-walk.mp4"]
    conditioning_masks = [
        [
            "inputs/2/masks/0.png",
            "inputs/2/masks/1.png",
            "inputs/2/masks/2.png",
            "inputs/2/masks/3.png",
        ]
    ]
    conditioning_start_frames = [0]

    device = get_device()
    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    session = LTXVideoSession(
        ckpt_path=ckpt_path,
        precision="bfloat16",
        text_encoder_model_name_or_path="PixArt-alpha/PixArt-XL-2-1024-MS",
        device=device,
        enhance_prompt=enhance_prompt,
        prompt_enhancer_image_caption_model_name_or_path="MiaoshouAI/Florence-2-large-PromptGen-v2.0",
        prompt_enhancer_llm_model_name_or_path="unsloth/Llama-3.2-3B-Instruct",
    )
    session.set_pipeline_args(
        num_inference_steps=num_inference_steps,
        generator=generator,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        prompt=prompt,
        negative_prompt=negative_prompt,
        enhance_prompt=enhance_prompt,
        conditioning_videos=conditioning_videos,
        conditioning_masks=conditioning_masks,
        conditioning_start_frames=conditioning_start_frames,
    )
    session.update_prompt()

    for i in range(3):
        session.update_conditioning(pop_latents=(None if i == 0 else 1))
        video = session.generate()
        if i == 0:
            session.set_pipeline_args(
                conditioning_videos=None,
                conditioning_masks=None,
                conditioning_start_frames=None,
            )

        # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
        video_np = video.permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        with imageio.get_writer(
            "outputs/output{}.mp4".format(i), fps=frame_rate
        ) as video:
            for frame in video_np:
                video.append_data(frame)
