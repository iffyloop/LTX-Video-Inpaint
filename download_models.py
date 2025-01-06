from pathlib import Path
from huggingface_hub import snapshot_download

MODELS_DIR = Path("/root/models")


def download_models():
    snapshot_download(
        "konakona/ltxvideo_q8",
        local_dir=(MODELS_DIR / "konakona--ltxvideo_q8"),
        local_dir_use_symlinks=False,
        repo_type="model",
    )
    snapshot_download(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        local_dir=(MODELS_DIR / "PixArt-alpha--PixArt-XL-2-1024-MS"),
        local_dir_use_symlinks=False,
        repo_type="model",
        allow_patterns=["tokenizer/**"],
    )
    snapshot_download(
        "Lightricks/LTX-Video",
        revision="af2c843632fc44500f8a468b50258d1d025cc1e0",
        local_dir=(MODELS_DIR / "Lightricks--LTX-Video"),
        local_dir_use_symlinks=False,
        repo_type="model",
        allow_patterns=["text_encoder/**", "vae/**", "scheduler/**"],
    )


if __name__ == "__main__":
    download_models()
