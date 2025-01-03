# LTX-Video-Inpaint

This is a fork of LTX-Video which supports inpainting. Please see the [original repository](https://github.com/Lightricks/LTX-Video) for more information.

## Usage

The `inference.py` script accepts the same text-to-video parameters as seen in the [original documentation for text-to-video generation](https://github.com/Lightricks/LTX-Video?tab=readme-ov-file#for-text-to-video-generation), but exposes two additional parameters:

- `--input_video_path`: Path to a video file which should be used for conditioning. The resolution and number of frames in this video _must exactly match_ the values of the `--width`, `--height`, and `--num_frames` arguments. If you are trying to extend a video, you can leave some empty frames at the end and fully mask them.
- `--input_mask_path`: Path to a video file which should be used as mask for inpainting. The resolution and number of frames in this video _must exactly match_ those of the `--input_video_path`, which in turn must match `--width`/`--height`/`--num_frames`. **Pixels which are white in this mask will be inpainted, while pixels which are black will be left alone**. Note that the mask is downsampled to latent space by averaging 32x32 blocks.

**Helpful reminder**: `--width`/`--height` _must_ both be divisible by 32 (e.g. 704x480), and `--num_frames` must be divisible by 8 + 1 (e.g. 9, 17, ..., 257).
