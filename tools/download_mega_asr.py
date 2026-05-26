#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mega-ASR Model Weights Downloader.
Downloads LoRA weights and environment quality router classification weights from Hugging Face.
Also downloads the Qwen3-ASR-1.7B base model if not already cached.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("download_mega_asr")

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    logger.error("huggingface_hub is not installed. Please install it with 'pip install huggingface_hub'.")
    sys.exit(1)


def download_file_safely(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download a single file from Hugging Face hub safely."""
    try:
        # Construct the target path
        target_path = local_dir / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading '{filename}' from Hugging Face repository '{repo_id}'...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )
        logger.info(f"Successfully downloaded to: {downloaded_path}")
        return Path(downloaded_path)
    except Exception as e:
        logger.error(f"Failed to download '{filename}' from '{repo_id}': {e}")
        raise


def download_mega_asr_weights(
    repo_id: str,
    lora_dir: str,
    router_dir: str,
    base_model_id: str,
    skip_base: bool = False
) -> bool:
    """Download Mega-ASR LoRA and router weights, and Qwen3-ASR base model."""
    success = True

    lora_path = Path(lora_dir)
    router_path = Path(router_dir)

    # Ensure directories exist
    lora_path.mkdir(parents=True, exist_ok=True)
    router_path.mkdir(parents=True, exist_ok=True)

    # 1. Download Qwen3-ASR Base Model
    if not skip_base:
        logger.info("=" * 60)
        logger.info(f"Step 1: Downloading Qwen3-ASR Base Model '{base_model_id}'...")
        logger.info("=" * 60)
        try:
            # We use snapshot_download to download the entire base model repository
            base_model_path = snapshot_download(repo_id=base_model_id)
            logger.info(f"Qwen3-ASR Base Model ready at cached location: {base_model_path}")
        except Exception as e:
            logger.error(f"Failed to download Qwen3-ASR Base Model '{base_model_id}': {e}")
            success = False
    else:
        logger.info("Skipping Qwen3-ASR base model download.")

    # 2. Download Mega-ASR LoRA Adapter weights
    logger.info("=" * 60)
    logger.info("Step 2: Downloading Mega-ASR LoRA adapter weights...")
    logger.info("=" * 60)
    try:
        temp_dir = Path("ckpt/Mega-ASR-temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading 'mega-asr-merged/adapter_model.safetensors' from Hugging Face repository '{repo_id}'...")
        downloaded_lora = hf_hub_download(
            repo_id=repo_id,
            filename="mega-asr-merged/adapter_model.safetensors",
            local_dir=str(temp_dir),
            local_dir_use_symlinks=False
        )
        final_lora_file = lora_path / "adapter_model.safetensors"
        final_lora_file.parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(final_lora_file):
            os.remove(final_lora_file)
        os.rename(downloaded_lora, final_lora_file)
        logger.info(f"Successfully placed LoRA weights to: {final_lora_file}")

        try:
            logger.info(f"Downloading 'mega-asr-merged/adapter_config.json' from Hugging Face repository '{repo_id}'...")
            downloaded_config = hf_hub_download(
                repo_id=repo_id,
                filename="mega-asr-merged/adapter_config.json",
                local_dir=str(temp_dir),
                local_dir_use_symlinks=False
            )
            final_config_file = lora_path / "adapter_config.json"
            if os.path.exists(final_config_file):
                os.remove(final_config_file)
            os.rename(downloaded_config, final_config_file)
            logger.info(f"Successfully placed LoRA config to: {final_config_file}")
        except Exception as e:
            logger.warning(f"LoRA adapter_config.json not found or download failed: {e}")
    except Exception as e:
        logger.error(f"Failed to download LoRA weights: {e}")
        success = False

    # 3. Download Audio Quality Router classification weights
    logger.info("=" * 60)
    logger.info("Step 3: Downloading Audio Quality Router weights...")
    logger.info("=" * 60)
    try:
        temp_dir = Path("ckpt/Mega-ASR-temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading 'audio_quality_router/best_acc_model.safetensors' from Hugging Face repository '{repo_id}'...")
        downloaded_router = hf_hub_download(
            repo_id=repo_id,
            filename="audio_quality_router/best_acc_model.safetensors",
            local_dir=str(temp_dir),
            local_dir_use_symlinks=False
        )
        final_router_file = router_path / "model.safetensors"
        final_router_file.parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(final_router_file):
            os.remove(final_router_file)
        os.rename(downloaded_router, final_router_file)
        logger.info(f"Successfully placed Router weights to: {final_router_file}")

        # Clean up temp directory
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    except Exception as e:
        logger.error(f"Failed to download Router weights: {e}")
        success = False

    logger.info("=" * 60)
    if success:
        logger.info("All Mega-ASR weights and components have been successfully prepared!")
        logger.info(f"LoRA weights: {lora_path}/adapter_model.safetensors")
        logger.info(f"Router weights: {router_path}/model.safetensors")
    else:
        logger.error("Some downloads failed. Please check the logs and retry.")
    logger.info("=" * 60)

    return success


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare model weights for Mega-ASR integration."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="zhifeixie/Mega-ASR",
        help="Hugging Face repository containing Mega-ASR weights (default: zhifeixie/Mega-ASR)"
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default="ckpt/Mega-ASR/lora",
        help="Local directory to store the LoRA adapter weights (default: ckpt/Mega-ASR/lora)"
    )
    parser.add_argument(
        "--router-dir",
        type=str,
        default="ckpt/Mega-ASR/router",
        help="Local directory to store the audio quality router weights (default: ckpt/Mega-ASR/router)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
        help="Hugging Face repo ID of the Qwen3-ASR base model (default: Qwen/Qwen3-ASR-1.7B)"
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip downloading the Qwen3-ASR base model"
    )

    args = parser.parse_args()

    success = download_mega_asr_weights(
        repo_id=args.repo_id,
        lora_dir=args.lora_dir,
        router_dir=args.router_dir,
        base_model_id=args.base_model,
        skip_base=args.skip_base
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
