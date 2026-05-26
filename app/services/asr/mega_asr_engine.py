# -*- coding: utf-8 -*-
"""
Mega-ASR (Robustness LoRA Enhanced) Engine.
实现了清华大学官方学术仓库（xzf-thu/Mega-ASR）高抗噪引擎与就地动态差值 LoRA 切换机制。
"""

from __future__ import annotations

import os
import json
import time
import math
import logging
import warnings
import threading
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
        x = x + self.pe[:, : x.size(1)]
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
# Mega-ASR Engine implementation (BaseASREngine)
# ==============================================================================

class MegaASREngine(BaseASREngine):
    """
    清华 Mega-ASR 离线高抗噪 ASR 引擎
    基于 PyTorch + LoRA Delta Switch 显存注入
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "auto",
        lora_path: Optional[str] = None,
        router_path: Optional[str] = None,
        degraded_threshold: float = 0.5,
        **_kwargs,
    ):
        if torch is None:
            raise RuntimeError("本引擎需要 PyTorch 环境支持，未检测到 torch 依赖库")

        from app.core.device import detect_device
        self._device = detect_device(device)
        self.model_id = "mega-asr-1.7b"
        self.model_path = model_path

        # 默认路径回退
        self.lora_path = lora_path or settings.MEGA_ASR_LORA_PATH
        self.router_path = router_path or settings.MEGA_ASR_ROUTER_PATH
        self.degraded_threshold = degraded_threshold

        # 并发推理锁，保证多路 ASR 显存就地安全修改
        self._infer_lock = threading.Lock()

        self.base_model = None
        self.tokenizer = None
        self.router = None
        self.lora_switch = None

        if not self._device.startswith("cuda"):
            raise DefaultServerErrorException(
                "Mega-ASR 引擎由于包含直接修改显存的操作，目前必须运行于 CUDA 显卡环境！"
            )

        self._load_all_components()

    def _load_all_components(self):
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer
            logger.info("正在初始化并载入 Qwen3-ASR 基座模型: %s", self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self._device)

            # 初始化音频路由器
            logger.info("正在初始化音频环境质量路由器: %s", self.router_path)
            self.router = AudioQualityRouter(
                model_path=self.router_path,
                device=self._device,
                threshold=self.degraded_threshold,
            )

            # 初始化 LoRA Delta Switch
            logger.info("正在计算并载入 LoRA 鲁棒增强权重差值: %s", self.lora_path)
            # 兼容：传入的 lora_path 如果 is_file，我们需要传递其所在的目录给 LoRADeltaSwitch
            lora_dir = os.path.dirname(self.lora_path) if self.lora_path.endswith(".safetensors") else self.lora_path
            self.lora_switch = LoRADeltaSwitch(
                base_model=self.base_model,
                adapter_dir=lora_dir,
                keep_delta_on_gpu=True,
            )
            logger.info("Mega-ASR 联合引擎所有组件全部加载并初始化就绪。")
        except Exception as e:
            logger.error("Mega-ASR 模型组件加载失败: %s", e)
            raise DefaultServerErrorException(f"Mega-ASR 模型组件加载失败: {e}")

    def is_model_loaded(self) -> bool:
        return self.base_model is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def supports_realtime(self) -> bool:
        return False

    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        enable_vad: bool = False,
        sample_rate: int = 16000,
    ) -> str:
        """单音频转写流程"""
        if not self.is_model_loaded():
            raise DefaultServerErrorException("Mega-ASR 引擎未正确装载")

        # 使用显存锁互斥操作
        with self._infer_lock:
            # 1. 路由器预判音频环境
            is_degraded = self.router.predict_is_degraded(audio_path)
            logger.info(
                "音频文件 '%s' 环境评估结果: %s",
                audio_path,
                "降级受污染 (Degraded)" if is_degraded else "纯净优良 (Clean)"
            )

            # 2. 动态就地合并权重差值
            if is_degraded:
                logger.debug("动态注入噪声鲁棒微调权重 ΔW...")
                self.lora_switch.merge()

            try:
                # 3. 执行 PyTorch 原生推理
                text = self._run_raw_inference(audio_path)
            finally:
                # 4. 回滚合并，恢复模型至基座状态
                if is_degraded:
                    logger.debug("动态扣除噪声鲁棒微调权重 ΔW 并恢复...")
                    self.lora_switch.unmerge()

            return normalize_asr_text(text, enable_itn=enable_itn)

    def _run_raw_inference(self, audio_path: str) -> str:
        """运行 PyTorch 模型的前向 ASR 推断"""
        y, _ = librosa.load(audio_path, sr=16000)
        inputs = self.tokenizer(y, sampling_rate=16000, return_tensors="pt").to(self._device)
        with torch.no_grad():
            generated_ids = self.base_model.generate(
                inputs.input_features,
                max_new_tokens=1024,
                generation_config=self.base_model.generation_config
            )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def transcribe_file_with_vad(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
        **_kwargs,
    ) -> ASRRawResult:
        text = self.transcribe_file(
            audio_path=audio_path,
            hotwords=hotwords,
            enable_punctuation=enable_punctuation,
            enable_itn=enable_itn,
            sample_rate=sample_rate,
        )
        return ASRRawResult(
            text=text,
            segments=[ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)]
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
        """批量多任务推理 - 高性能分组机制"""
        if not segments:
            return []

        # 双向倒排索引重映射准备
        clean_idx_map = []
        degraded_idx_map = []

        # 1. 对整个 batch 的所有切片进行路由器并行判定并归类
        for idx, seg in enumerate(segments):
            if not seg.temp_file:
                continue
            is_degraded = self.router.predict_is_degraded(seg.temp_file)
            if is_degraded:
                degraded_idx_map.append((idx, seg.temp_file))
            else:
                clean_idx_map.append((idx, seg.temp_file))

        logger.info(
            "音频切片批量分流完成: 纯净良质(Clean)数量=%d, 降级抗噪(Degraded)数量=%d",
            len(clean_idx_map),
            len(degraded_idx_map),
        )

        results_dict: Dict[int, str] = {}

        # 锁显存
        with self._infer_lock:
            # 2. 第一步：在 Base 状态（无 LoRA）下批量推理 Clean 组
            if clean_idx_map:
                logger.debug("开始执行 Clean 音频分流的并行推断...")
                for orig_idx, path in clean_idx_map:
                    try:
                        results_dict[orig_idx] = self._run_raw_inference(path)
                    except Exception as exc:
                        logger.error("Clean 音频切片推理失败 orig_idx=%d: %s", orig_idx, exc)
                        results_dict[orig_idx] = ""

            # 3. 第二步：一键加装 LoRA，并批量推理 Degraded 组
            if degraded_idx_map:
                logger.debug("切换模型权重 (LoRA Delta Switch Merge)...")
                self.lora_switch.merge()
                try:
                    logger.debug("开始执行 Degraded 音频分流的鲁棒增强型并行推断...")
                    for orig_idx, path in degraded_idx_map:
                        try:
                            results_dict[orig_idx] = self._run_raw_inference(path)
                        except Exception as exc:
                            logger.error("Degraded 音频切片推理失败 orig_idx=%d: %s", orig_idx, exc)
                            results_dict[orig_idx] = ""
                finally:
                    # 4. 第三步：推断完毕卸载 LoRA，恢复基座
                    logger.debug("回滚并释放模型权重 (LoRA Delta Switch Unmerge)...")
                    self.lora_switch.unmerge()

        # 5. 反重组序列拼装
        final_results = []
        for idx, seg in enumerate(segments):
            raw_text = results_dict.get(idx, "")
            processed_text = normalize_asr_text(raw_text, enable_itn=enable_itn)
            final_results.append(
                ASRSegmentResult(
                    text=processed_text,
                    start_time=getattr(seg, "start_sec", 0.0),
                    end_time=getattr(seg, "end_sec", 0.0),
                    speaker_id=getattr(seg, "speaker_id", None)
                )
            )

        return final_results


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
