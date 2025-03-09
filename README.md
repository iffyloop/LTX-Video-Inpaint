# LTX-Video-Inpaint

## Installation

```sh
git clone https://github.com/iffyloop/LTX-Video-Inpaint.git
cd LTX-Video-Inpaint

python -m venv env
source env/bin/activate
python -m pip install -e .\[inference-script\]
```

Then download the model from [HuggingFace](https://huggingface.co/Lightricks/LTX-Video/tree/main) (latest version is `ltx-video-2b-v0.9.5.safetensors` at this time of writing).

## Inference

```sh
python inference.py --ckpt_path "/path/to/ltx-video-2b-v0.9.5.safetensors" --prompt "PROMPT" --conditioning_media_paths "VIDEO.mp4" --conditioning_mask_paths "MASKS_FOLDER" --conditioning_start_frames 0 --height HEIGHT --width WIDTH --num_frames NUM_FRAMES --seed SEED
```

### Notes on Inference

- We only support conditioning start from first frame (`--conditioning_start_frames 0`)
- The masks folder (`--conditioning_mask_paths`) should contain PNG files according to this pattern:
  - `0.png` contains the mask for the first frame of the video
  - `1.png` contains the mask for frames 2-9
  - `2.png` contains the mask for frames 10-17
  - Pattern continues for every group of 8 frames
- Masks should be of size (video width / 32, video height / 32) - for example, if your video dimensions are 704x480 pixels, then each mask image should be 22x15 pixels in size. Each pixel of the mask corresponds to one group of 32x32 pixels in the video.
- Regions marked black in the mask are inpainted, while regions marked white are NOT inpainted. **This is opposite of many image generation tools.**
