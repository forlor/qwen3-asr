# -*- coding: utf-8 -*-
"""
Mega-ASR (Robustness LoRA Enhanced) Engine.
实现了清华大学官方学术仓库（xzf-thu/Mega-ASR）高抗噪引擎与就地动态差值 LoRA 切换机制。
"""

from __future__ import annotations

import gc
import os
import json
import shutil
import time
import math
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Any, Dict

import numpy as np

# 延迟安全导入依赖项，防止非 GPU/本地测试环境导入崩溃
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import safe_open
except ImportError:
    safe_load_file = None
    safe_open = None

from .engines.base import BaseASREngine, ASRRawResult, ASRSegmentResult
from .qwen3_vllm import Qwen3VLLMBackend, is_vllm_available
from ...core.exceptions import DefaultServerErrorException
from ...core.config import settings
from ...utils.text_processing import normalize_asr_text

logger = logging.getLogger(__name__)

# 定义动态基类，兼容非 Torch 导入测试环境
_BaseModule = nn.Module if nn is not None else object


# ==============================================================================
# Audio Quality Router Neural Network Definition (From official xzf-thu/Mega-ASR)
# ==============================================================================

class LogMelSpectrogram(_BaseModule):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
    ) -> None:
        super().__init__()
        if torchaudio is None:
            raise ImportError("未检测到 torchaudio，LogMelSpectrogram 无法工作")
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(waveform)
        log_mel = torch.clamp(mel, min=1e-10).log10()
        return (log_mel + 4.0) / 4.0


class PositionalEncoding(_BaseModule):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # 动态扩展 Positional Encoding，支持超长音频切片，防止维度不匹配报错
            pe = torch.zeros(seq_len, self.pe.size(2), device=self.pe.device, dtype=self.pe.dtype)
            position = torch.arange(0, seq_len, dtype=torch.float, device=self.pe.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.pe.size(2), 2, device=self.pe.device).float() * (-math.log(10000.0) / self.pe.size(2))
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class AttentionPooling(_BaseModule):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        weights = self.query(x).squeeze(-1)

        if mask is not None:
            weights = weights.masked_fill(~mask, float("-inf"))

        weights = F.softmax(weights, dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class ConvFrontend(_BaseModule):
    def __init__(self, n_mels: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)


class AudioQualityClassifier(_BaseModule):
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 192,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 3000,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.downsample_rate = 4
        self.frontend = ConvFrontend(n_mels, d_model, dropout)
        self.pos_encoder = PositionalEncoding(d_model, max_len // 4 + 100, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            norm=nn.LayerNorm(d_model),
        )

        self.pooling = AttentionPooling(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

    def forward(
        self,
        mels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.frontend(mels)
        time_steps = x.shape[1]

        if mask is not None:
            mask = mask[:, :: self.downsample_rate]
            if mask.shape[1] > time_steps:
                mask = mask[:, :time_steps]
            elif mask.shape[1] < time_steps:
                pad = torch.ones(
                    mask.shape[0],
                    time_steps - mask.shape[1],
                    device=mask.device,
                    dtype=mask.dtype,
                )
                mask = torch.cat([mask, pad], dim=1)

        x = self.pos_encoder(x)
        src_key_padding_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.pooling(x, mask)
        return self.classifier(x)


def create_audio_quality_model(config: dict) -> torch.nn.Module:
    return AudioQualityClassifier(
        n_mels=config.get("n_mels", 80),
        d_model=config.get("d_model", 192),
        nhead=config.get("nhead", 4),
        dim_feedforward=config.get("dim_feedforward", 512),
        dropout=config.get("dropout", 0.1),
        max_len=config.get("max_len", 3000),
        num_classes=config.get("num_classes", 2),
    )


# ==============================================================================
# Audio Quality Router (Official Wrapper with Log-Mel Feature extraction)
# ==============================================================================

class AudioQualityRouter:
    """音频质量评估神经网络路由器（Log-Mel Spectrogram + Transformer Classifier）"""

    def __init__(self, model_path: str, device: str, threshold: float = 0.5):
        if torch is None:
            raise ImportError("未检测到 PyTorch，无法初始化 AudioQualityRouter")
        if librosa is None:
            raise ImportError("未检测到 librosa 库，无法提取音频")
        if safe_open is None or safe_load_file is None:
            raise ImportError("未检测到 safetensors 库，无法加载权重")

        self.device = device
        self.model_path = model_path
        self.threshold = threshold
        self.sample_rate = 16000

        self._load_network()

    def _load_network(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"分类路由器权重路径不存在: {self.model_path}")

            # 提取 metadata 配置
            with safe_open(self.model_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
            checkpoint_config = json.loads(metadata.get("config", "{}")) if metadata else {}
            config = checkpoint_config.get("model", {})
            state_dict = safe_load_file(self.model_path, device=self.device)

            self.model = create_audio_quality_model(config)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.mel_extractor = LogMelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=config.get("n_mels", 80),
            ).to(self.device)
            self.mel_extractor.eval()

            logger.info("音频环境质量分类路由器加载并初始化成功: %s", self.model_path)
        except Exception as e:
            logger.error("分类路由器加载失败: %s", e)
            raise RuntimeError(f"分类路由器加载失败: {e}")

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        waveform = torch.from_numpy(y).float().unsqueeze(0)
        return waveform.to(self.device)

    def predict_is_degraded(self, audio_path: str, threshold: Optional[float] = None) -> bool:
        """预测单条音频环境是否降级(嘈杂/受污染)"""
        th = threshold if threshold is not None else self.threshold
        try:
            with torch.no_grad():
                waveform = self._load_audio(audio_path)
                mel = self.mel_extractor(waveform)
                mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0)

                logits = self.model(mel, mask=None)
                probs = torch.softmax(logits, dim=-1)
                degraded_prob = float(probs[0, 1].item())
                return degraded_prob >= th
        except Exception as e:
            logger.warning(
                "路由器评估出错 '%s': %s, 默认回退使用基座模式 (Clean)",
                audio_path,
                e,
            )
            return False

    def predict(self, audio_path: str, threshold: Optional[float] = None) -> Dict[str, Any]:
        """预测音频质量，返回包含标签和概率的字典。

        Returns:
            dict: {"label": "clean"|"degraded", "degraded_prob": float}
        """
        th = threshold if threshold is not None else self.threshold
        try:
            with torch.no_grad():
                waveform = self._load_audio(audio_path)
                mel = self.mel_extractor(waveform)
                mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0)

                logits = self.model(mel, mask=None)
                probs = torch.softmax(logits, dim=-1)
                degraded_prob = float(probs[0, 1].item())
                label = "degraded" if degraded_prob >= th else "clean"
                return {"label": label, "degraded_prob": round(degraded_prob, 4)}
        except Exception as e:
            logger.warning("音频质量评估出错 '%s': %s", audio_path, e)
            return {"label": "unknown", "degraded_prob": -1.0}


# ==============================================================================
# LoRA Delta Switch Controller (Official xzf-thu/Mega-ASR lora_switch.py)
# ==============================================================================

class LoRADeltaSwitch:
    """LoRA 偏差矩阵安全加载与就地合并控制器（完全支持 mega_lora_blocks.json 分块逻辑）"""

    def __init__(self, base_model, adapter_dir: str, keep_delta_on_gpu: bool = True) -> None:
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.keep_delta_on_gpu = keep_delta_on_gpu
        self.items: list[dict[str, Any]] = []
        self.active = False
        self._load_and_add_adapter()

    def _load_adapter_state(self) -> dict[str, torch.Tensor]:
        safetensors_path = os.path.join(self.adapter_dir, "adapter_model.safetensors")
        bin_path = os.path.join(self.adapter_dir, "adapter_model.bin")

        if os.path.exists(safetensors_path):
            return safe_load_file(safetensors_path)
        if os.path.exists(bin_path):
            return torch.load(bin_path, map_location="cpu")
        raise FileNotFoundError(f"在 {self.adapter_dir} 下未找到 LoRA 权重文件")

    def _load_adapter_config(self) -> dict[str, Any]:
        config_path = os.path.join(self.adapter_dir, "adapter_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_adapter_blocks(self) -> dict[str, Any]:
        blocks_path = os.path.join(self.adapter_dir, "mega_lora_blocks.json")
        if not os.path.exists(blocks_path):
            return {}
        with open(blocks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _normalize_module_name(name: str) -> str:
        for prefix in ("base_model.model.",):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        if name.startswith("thinker.layers."):
            name = name.replace("thinker.layers.", "thinker.model.layers.", 1)
        return name

    @staticmethod
    def _module_name_candidates(name: str) -> list[str]:
        candidates = [name]
        if name.startswith("model."):
            candidates.append(name[len("model.") :])
        if name.startswith("thinker.layers."):
            candidates.append(name.replace("thinker.layers.", "thinker.model.layers.", 1))
        if name.startswith("thinker.model."):
            candidates.append(name.replace("thinker.model.", "thinker.", 1))
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _raw_module_name(key: str, marker: str) -> str:
        name = key.split(marker)[0]
        for prefix in ("base_model.model.", "model."):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    def _split_lora_key(self, key: str) -> tuple[str | None, str | None, str | None]:
        raw_key = key
        key = self._normalize_module_name(key)

        for marker in (".lora_A.", ".lora_B."):
            if marker in key:
                module_name = key.split(marker)[0]
                raw_module_name = self._raw_module_name(raw_key, marker)
                kind = "A" if marker == ".lora_A." else "B"
                return module_name, raw_module_name, kind
        return None, None, None

    def _load_and_add_adapter(self) -> None:
        config = self._load_adapter_config()
        state = self._load_adapter_state()
        blocks = self._load_adapter_blocks()

        lora_alpha = config.get("lora_alpha", 1)
        rank = config.get("r")
        alpha_pattern = config.get("alpha_pattern") or {}
        rank_pattern = config.get("rank_pattern") or {}
        fan_in_fan_out = bool(config.get("fan_in_fan_out", False))

        module_dict = dict(self.base_model.named_modules())
        grouped: dict[str, dict[str, torch.Tensor]] = {}

        for key, tensor in state.items():
            module_name, raw_module_name, kind = self._split_lora_key(key)
            if module_name is None or raw_module_name is None or kind is None:
                continue

            matched_name = None
            for candidate in self._module_name_candidates(module_name):
                if candidate in module_dict:
                    matched_name = candidate
                    break

            target_name = matched_name or module_name
            group_key = f"{target_name}\0{raw_module_name}"
            item = grouped.setdefault(
                group_key,
                {
                    "target_module_name": target_name,
                    "raw_module_name": raw_module_name,
                },
            )
            item[kind] = tensor.cpu()

        loaded = 0
        missing = []

        for pair in grouped.values():
            if "A" not in pair or "B" not in pair:
                continue
            module_name = pair["target_module_name"]
            raw_module_name = pair["raw_module_name"]
            if module_name not in module_dict:
                missing.append(module_name)
                continue

            module = module_dict[module_name]
            if not hasattr(module, "weight"):
                missing.append(module_name)
                continue

            weight = module.weight
            a_matrix = pair["A"].to(device=weight.device, dtype=torch.float32)
            b_matrix = pair["B"].to(device=weight.device, dtype=torch.float32)
            module_blocks = blocks.get(raw_module_name) or blocks.get(module_name)

            if module_blocks:
                deltas = []
                for block in module_blocks:
                    start = int(block["start"])
                    end = int(block["end"])
                    block_rank = int(block.get("rank", end - start))
                    block_alpha = int(block.get("alpha", block_rank))
                    delta = torch.matmul(b_matrix[:, start:end], a_matrix[start:end])
                    delta = delta * (float(block_alpha) / float(block_rank))
                    if fan_in_fan_out:
                        delta = delta.T
                    deltas.append(delta)
            else:
                adapter_rank = rank_pattern.get(raw_module_name, rank_pattern.get(module_name, rank))
                if adapter_rank is None:
                    adapter_rank = a_matrix.shape[0]
                adapter_alpha = alpha_pattern.get(
                    raw_module_name,
                    adapter_alpha := alpha_pattern.get(module_name, lora_alpha),
                )
                scaling = float(adapter_alpha) / float(adapter_rank)
                delta = torch.matmul(b_matrix, a_matrix) * scaling
                if fan_in_fan_out:
                    delta = delta.T
                deltas = [delta]

            for delta in deltas:
                if delta.shape != weight.shape:
                    try:
                        delta = delta.reshape(weight.shape)
                    except Exception:
                        missing.append(
                            f"{module_name}: delta shape {tuple(delta.shape)} != "
                            f"weight shape {tuple(weight.shape)}"
                        )
                        continue

                delta = delta.to(dtype=weight.dtype)
                if self.keep_delta_on_gpu:
                    delta = delta.to(device=weight.device)
                else:
                    delta = delta.cpu()

                self.items.append(
                    {
                        "module_name": module_name,
                        "weight": weight,
                        "delta": delta,
                    }
                )
                loaded += 1

        if missing:
            warnings.warn(
                f"LoRA adapter loaded {loaded} deltas, "
                f"missing {len(missing)} modules. Examples: {missing[:5]}",
                stacklevel=2,
            )
        logger.info("Mega-ASR LoRA Delta 预计算就绪，共编译了 %d 个权重的适配 Delta 矩阵", loaded)

    def merge(self):
        """就地叠加差值"""
        if self.active:
            return
        with torch.no_grad():
            for item in self.items:
                weight = item["weight"]
                delta = item["delta"]
                if delta.device != weight.device:
                    delta = delta.to(device=weight.device)
                weight.data.add_(delta, alpha=1.0)
        self.active = True

    def unmerge(self):
        """就地扣除差值"""
        if not self.active:
            return
        with torch.no_grad():
            for item in self.items:
                weight = item["weight"]
                delta = item["delta"]
                if delta.device != weight.device:
                    delta = delta.to(device=weight.device)
                weight.data.add_(delta, alpha=-1.0)
        self.active = False


# ==============================================================================
# Mega-ASR Engine implementation (BaseASREngine) — vLLM + LoRA Materialization
# ==============================================================================

@dataclass
class MegaASRStreamingState:
    internal_state: Any
    chunk_size_sec: float = 2.0
    unfixed_chunk_num: int = 2
    unfixed_token_num: int = 5
    max_new_tokens: int = 32
    language: Optional[str] = None
    chunk_count: int = 0
    last_text: str = ""
    last_language: str = ""


class MegaASREngine(BaseASREngine):
    """
    清华 Mega-ASR 离线高抗噪 ASR 引擎
    CUDA 环境使用 vLLM 后端 + LoRA 物化（预合并）方案
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "auto",
        lora_path: Optional[str] = None,
        router_path: Optional[str] = None,
        degraded_threshold: float = 0.5,
        max_inference_batch_size: int = 16,
        max_new_tokens: int = 1024,
        max_model_len: Optional[int] = None,
        **_kwargs,
    ):
        if torch is None:
            raise RuntimeError("Mega-ASR requires PyTorch (needed for LoRA materialization)")

        from app.core.device import detect_device

        self._device = detect_device(device)
        self.model_id = "mega-asr-1.7b"
        self.model_path = model_path

        self.lora_path = lora_path or settings.MEGA_ASR_LORA_PATH
        self.router_path = router_path or settings.MEGA_ASR_ROUTER_PATH
        self.degraded_threshold = degraded_threshold

        self._backend = self._select_backend()
        self.model: Optional[Qwen3VLLMBackend] = None

        try:
            if self._backend == "vllm":
                materialized_path = self._materialize_checkpoint()
                self.model = self._load_vllm(
                    materialized_path,
                    max_inference_batch_size=max_inference_batch_size,
                    max_new_tokens=max_new_tokens,
                    max_model_len=max_model_len,
                )
            logger.info("Mega-ASR engine loaded successfully with backend=%s", self._backend)
        except Exception as e:
            logger.error("Mega-ASR engine load failed: %s", e)
            raise DefaultServerErrorException(f"Mega-ASR engine load failed: {e}")

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _select_backend(self) -> str:
        if self._device.startswith("cuda"):
            if not is_vllm_available():
                raise DefaultServerErrorException(
                    "Mega-ASR on CUDA requires vLLM. "
                    "Install with: pip install 'vllm[audio]==0.19.0'"
                )
            return "vllm"
        raise DefaultServerErrorException(
            f"Mega-ASR is not available on device '{self._device}'. Only CUDA + vLLM is supported."
        )

    # ------------------------------------------------------------------
    # LoRA Materialization (one-time pre-merge)
    # ------------------------------------------------------------------

    _MATERIALIZE_MARKER = "mega_asr_materialized.json"

    def _is_materialized_fresh(self, output_dir: Path) -> bool:
        """Check if the materialized checkpoint is fresh (matches current config)."""
        marker_path = output_dir / self._MATERIALIZE_MARKER
        if not marker_path.is_file() or not (output_dir / "config.json").is_file():
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        expected = self._marker_payload(output_dir)
        return marker == expected

    def _marker_payload(self, output_dir: Path) -> dict[str, Any]:
        """Build a fingerprint dict for freshness checking (matches official Mega-ASR)."""
        lora_dir = (
            os.path.dirname(self.lora_path)
            if self.lora_path.endswith(".safetensors")
            else self.lora_path
        )
        base_path = Path(self.model_path).expanduser()
        adapter_path = Path(lora_dir).expanduser()
        return {
            "base_model_path": str(base_path),
            "lora_dir": str(adapter_path),
            "base_config_mtime": self._path_mtime(base_path / "config.json"),
            "adapter_config_mtime": self._path_mtime(adapter_path / "adapter_config.json"),
            "adapter_safetensors_mtime": self._path_mtime(adapter_path / "adapter_model.safetensors"),
            "adapter_bin_mtime": self._path_mtime(adapter_path / "adapter_model.bin"),
            "mega_lora_blocks_mtime": self._path_mtime(adapter_path / "mega_lora_blocks.json"),
        }

    @staticmethod
    def _path_mtime(path: Path) -> float | None:
        return path.stat().st_mtime if path.exists() else None

    def _materialize_checkpoint(self) -> str:
        """Merge LoRA deltas into base model weights and save to disk.

        Follows the official xzf-thu/Mega-ASR materialize_lora.py approach:
        load base → apply LoRA delta → save merged model + processor.
        """
        output_dir = Path(settings.MEGA_ASR_VLLM_MATERIALIZED_PATH)

        # Use file locking to prevent concurrent workers from racing.
        try:
            from filelock import FileLock
        except ImportError:
            FileLock = None

        lock_path = output_dir.with_name(output_dir.name + ".lock")
        lock = FileLock(str(lock_path)) if FileLock else None

        def _do_materialize() -> str:
            # Freshness check inside lock to avoid TOCTOU races.
            if (
                not settings.MEGA_ASR_FORCE_REMATERIALIZE
                and self._is_materialized_fresh(output_dir)
            ):
                logger.info("Materialized checkpoint is fresh: %s", output_dir)
                return str(output_dir)

            if settings.MEGA_ASR_FORCE_REMATERIALIZE:
                logger.info("Force re-materialization requested")

            logger.info("=" * 60)
            logger.info("Starting LoRA materialization (one-time pre-merge)...")
            logger.info("Base model: %s", self.model_path)
            logger.info("LoRA adapter: %s", self.lora_path)
            logger.info("Output: %s", output_dir)
            logger.info("=" * 60)

            from qwen_asr import Qwen3ASRModel

            # Match official defaults: bfloat16 on CUDA, float32 on CPU.
            mat_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            mat_dtype = torch.bfloat16 if mat_device != "cpu" else torch.float32

            # 1. Load base model with minimal params (official uses 1, 1).
            logger.info("Step 1/5: Loading base model on %s (%s)...", mat_device, mat_dtype)
            base_model = Qwen3ASRModel.from_pretrained(
                self.model_path,
                dtype=mat_dtype,
                device_map=mat_device,
                max_inference_batch_size=1,
                max_new_tokens=1,
            )

            # 2. Apply LoRA via LoRADeltaSwitch
            logger.info("Step 2/5: Applying LoRA delta weights...")
            lora_dir = (
                os.path.dirname(self.lora_path)
                if self.lora_path.endswith(".safetensors")
                else self.lora_path
            )
            lora_switch = LoRADeltaSwitch(
                base_model=base_model.model,
                adapter_dir=lora_dir,
                keep_delta_on_gpu=(mat_device.startswith("cuda")),
            )
            lora_switch.merge()
            logger.info(
                "LoRA merged successfully (%d weight deltas applied)",
                len(lora_switch.items),
            )

            # 3. Clean output directory, then save merged model + processor.
            logger.info("Step 3/5: Saving materialized checkpoint to %s...", output_dir)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            base_model.model.save_pretrained(
                str(output_dir),
                safe_serialization=True,
                max_shard_size="2GB",
            )
            base_model.processor.save_pretrained(str(output_dir))

            # 4. Write freshness marker.
            marker = self._marker_payload(output_dir)
            (output_dir / self._MATERIALIZE_MARKER).write_text(
                json.dumps(marker, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            # 5. Cleanup.
            logger.info("Step 4/5: Releasing memory...")
            del lora_switch
            del base_model
            gc.collect()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("=" * 60)
            logger.info("LoRA materialization complete! Checkpoint saved to: %s", output_dir)
            logger.info("=" * 60)
            return str(output_dir)

        if lock:
            with lock:
                return _do_materialize()
        return _do_materialize()

    # ------------------------------------------------------------------
    # vLLM backend loading
    # ------------------------------------------------------------------

    def _load_vllm(
        self,
        model_path: str,
        max_inference_batch_size: int,
        max_new_tokens: int,
        max_model_len: Optional[int],
    ) -> Qwen3VLLMBackend:
        from .qwen3_engine import calculate_gpu_memory_utilization

        gpu_memory_utilization = calculate_gpu_memory_utilization(model_path)
        logger.info(
            "Loading Mega-ASR (vLLM): %s, gpu_memory_utilization=%s",
            model_path,
            gpu_memory_utilization,
        )
        return Qwen3VLLMBackend(
            model_path=model_path,
            forced_aligner_path=None,
            gpu_memory_utilization=gpu_memory_utilization,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
        )

    # ------------------------------------------------------------------
    # Engine interface
    # ------------------------------------------------------------------

    def is_model_loaded(self) -> bool:
        return self.model is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def supports_realtime(self) -> bool:
        return self._backend == "vllm"

    # ------------------------------------------------------------------
    # Offline transcription
    # ------------------------------------------------------------------

    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        enable_vad: bool = False,
        sample_rate: int = 16000,
    ) -> str:
        if not self.is_model_loaded():
            raise DefaultServerErrorException("Mega-ASR engine not loaded")
        if self._backend == "vllm":
            return self.model.transcribe_text(
                audio_path,
                context=hotwords or "",
                enable_itn=enable_itn,
            )
        raise DefaultServerErrorException(
            f"Mega-ASR backend={self._backend} does not support transcription"
        )

    def transcribe_file_with_vad(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
        **kwargs,
    ) -> ASRRawResult:
        if not self.is_model_loaded():
            raise DefaultServerErrorException("Mega-ASR engine not loaded")
        if self._backend == "vllm":
            return self.model.transcribe_raw(
                audio_path=audio_path,
                context=hotwords or "",
                language=kwargs.get("language"),
                word_timestamps=kwargs.get("word_timestamps", False),
                enable_itn=enable_itn,
            )
        raise DefaultServerErrorException(
            f"Mega-ASR backend={self._backend} does not support VAD transcription"
        )

    def _transcribe_batch(
        self,
        segments: List[Any],
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        sample_rate: int = 16000,
        word_timestamps: bool = False,
    ) -> List[ASRSegmentResult]:
        if not segments:
            return []
        if not self.is_model_loaded():
            raise DefaultServerErrorException("Mega-ASR engine not loaded")

        if self._backend == "vllm":
            output = [
                ASRSegmentResult(text="", start_time=0.0, end_time=0.0)
                for _ in segments
            ]
            valid: List[tuple[int, Any]] = []
            for idx, seg in enumerate(segments):
                temp_file = getattr(seg, "temp_file", None)
                if temp_file and os.path.exists(temp_file):
                    valid.append((idx, seg))
                else:
                    logger.warning(
                        "Mega-ASR batch segment invalid: segment=%d, file=%s",
                        idx + 1,
                        temp_file,
                    )
            if not valid:
                return output

            # Per-segment error handling: a single corrupted audio should not
            # abort the entire batch.
            valid_paths = [seg.temp_file for _, seg in valid]
            try:
                vllm_results = self.model.transcribe_batch(
                    valid_paths,
                    context=hotwords or "",
                    word_timestamps=word_timestamps,
                    enable_itn=enable_itn,
                )
            except Exception as exc:
                logger.error("Mega-ASR vLLM batch failed, falling back to per-segment: %s", exc)
                vllm_results = []
                for _idx, seg in valid:
                    try:
                        result = self.model.transcribe_batch(
                            [seg.temp_file],
                            context=hotwords or "",
                            word_timestamps=word_timestamps,
                            enable_itn=enable_itn,
                        )
                        vllm_results.append(result[0])
                    except Exception as seg_exc:
                        logger.error(
                            "Mega-ASR segment %d failed: %s",
                            _idx + 1,
                            seg_exc,
                        )
                        vllm_results.append(
                            ASRSegmentResult(text="", start_time=0.0, end_time=0.0)
                        )
            for (idx, seg), result in zip(valid, vllm_results):
                output[idx] = ASRSegmentResult(
                    text=result.text,
                    start_time=round(seg.start_sec, 2),
                    end_time=round(seg.end_sec, 2),
                    speaker_id=getattr(seg, "speaker_id", None),
                    word_tokens=result.word_tokens if word_timestamps else None,
                )
            return output

        raise DefaultServerErrorException(
            f"Mega-ASR backend={self._backend} does not support batch transcription"
        )

    # ------------------------------------------------------------------
    # Streaming (vLLM only)
    # ------------------------------------------------------------------

    def init_streaming_state(
        self,
        context: str = "",
        language: Optional[str] = None,
        **kwargs,
    ) -> MegaASRStreamingState:
        if self._backend != "vllm":
            raise DefaultServerErrorException(
                f"Mega-ASR backend={self._backend} does not support streaming"
            )
        streaming_state = self.model.init_streaming_state(
            context=context, language=language, **kwargs
        )
        return MegaASRStreamingState(
            internal_state=streaming_state,
            chunk_size_sec=float(kwargs.get("chunk_size_sec", 2.0)),
            unfixed_chunk_num=int(kwargs.get("unfixed_chunk_num", 2)),
            unfixed_token_num=int(kwargs.get("unfixed_token_num", 5)),
            max_new_tokens=int(kwargs.get("max_new_tokens", 32)),
            language=language,
            chunk_count=int(getattr(streaming_state, "chunk_id", 0)),
            last_text=str(getattr(streaming_state, "text", "") or ""),
            last_language=str(getattr(streaming_state, "language", "") or ""),
        )

    def streaming_transcribe(
        self,
        pcm16k: np.ndarray,
        state: MegaASRStreamingState,
    ) -> MegaASRStreamingState:
        if self._backend != "vllm":
            raise DefaultServerErrorException(
                f"Mega-ASR backend={self._backend} does not support streaming"
            )
        pcm = pcm16k.astype(np.float32) / (
            32768.0 if pcm16k.dtype == np.int16 else 1.0
        )
        streaming_state = self.model.feed_stream(pcm, state.internal_state)
        state.internal_state = streaming_state
        state.chunk_count = int(
            getattr(streaming_state, "chunk_id", state.chunk_count)
        )
        state.last_text = str(
            getattr(streaming_state, "text", "") or ""
        )
        state.last_language = str(
            getattr(streaming_state, "language", "") or ""
        )
        return state

    def finish_streaming_transcribe(
        self,
        state: MegaASRStreamingState,
    ) -> MegaASRStreamingState:
        if self._backend != "vllm":
            raise DefaultServerErrorException(
                f"Mega-ASR backend={self._backend} does not support streaming"
            )
        streaming_state = self.model.finish_stream(state.internal_state)
        state.internal_state = streaming_state
        state.chunk_count = int(
            getattr(streaming_state, "chunk_id", state.chunk_count)
        )
        state.last_text = str(
            getattr(streaming_state, "text", "") or ""
        )
        state.last_language = str(
            getattr(streaming_state, "language", "") or ""
        )
        return state


def _register_mega_asr_engine(register_func, _declared_entry_cls):
    """引擎自动初始化绑定"""
    def _create(config):
        extra = {k: v for k, v in config.extra_kwargs.items() if v is not None}
        model_id = config.models.get("offline")
        return MegaASREngine(
            model_path=model_id,
            device=settings.DEVICE,
            lora_path=settings.MEGA_ASR_LORA_PATH,
            router_path=settings.MEGA_ASR_ROUTER_PATH,
            degraded_threshold=settings.MEGA_ASR_DEGRADED_THRESHOLD,
            **extra
        )

    register_func("mega_asr", _create)
