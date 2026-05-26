# -*- coding: utf-8 -*-
"""
Mega-ASR (Robustness LoRA Enhanced) Engine.
实现高内聚的 PyTorch 离线 ASR 与动态批分组算法。
"""

import logging
import os
import threading
from typing import Optional, List, Any, Dict

import numpy as np

# 延迟安全导入依赖项，防止非 GPU/未安装依赖环境在导入时崩溃
try:
    import torch
except ImportError:
    torch = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None

from .engines.base import BaseASREngine, ASRRawResult, ASRSegmentResult, WordToken
from ...core.exceptions import DefaultServerErrorException
from ...core.config import settings
from ...utils.text_processing import normalize_asr_text

logger = logging.getLogger(__name__)


class AudioQualityRouter:
    """音频质量评估神经网络路由器（Log-Mel Spectrogram + CNN Classifier）"""

    def __init__(self, model_path: str, device: str):
        if torch is None:
            raise ImportError("未检测到 PyTorch 库，无法初始化 AudioQualityRouter")
        if librosa is None:
            raise ImportError("未检测到 librosa 库，无法提取梅尔特征谱")
        if load_file is None:
            raise ImportError("未检测到 safetensors 库，无法读取路由器权重")

        self.device = device
        self.model_path = model_path
        self._load_network()

    def _load_network(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"分类路由器权重路径不存在: {self.model_path}")

            # 动态加载分类器 safetensors 权重
            self.weights = load_file(self.model_path)
            # 初始化层参数并搬移至 GPU / CPU 设备
            self.w_conv1 = self.weights["conv1.weight"].to(self.device)
            self.b_conv1 = self.weights["conv1.bias"].to(self.device)
            self.w_conv2 = self.weights["conv2.weight"].to(self.device)
            self.b_conv2 = self.weights["conv2.bias"].to(self.device)
            self.w_fc = self.weights["fc.weight"].to(self.device)
            self.b_fc = self.weights["fc.bias"].to(self.device)
            logger.info("音频质量分类路由器权重加载成功: %s", self.model_path)
        except Exception as e:
            logger.error("分类路由器权重载入失败: %s", e)
            raise RuntimeError(f"分类路由器权重载入失败: {e}")

    def _extract_mel(self, audio_path: str) -> "torch.Tensor":
        """提取标准的 Log-Mel 特征"""
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        # N_FFT=1024, Hop=256, Mel=80
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80, fmin=0, fmax=8000
        )
        log_mel = np.log(np.maximum(mel_spec, 1e-5))
        # 增加 Batch 与 Channel 维度: [1, 1, 80, T]
        return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

    def predict_is_degraded(self, audio_path: str, threshold: float = 0.5) -> bool:
        """预测单条音频是否降级"""
        try:
            with torch.no_grad():
                x = self._extract_mel(audio_path)
                # Conv1 -> Relu
                x = torch.nn.functional.conv2d(x, self.w_conv1, self.b_conv1, padding=1)
                x = torch.relu(x)
                # MaxPool2d
                x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
                # Conv2 -> Relu
                x = torch.nn.functional.conv2d(x, self.w_conv2, self.b_conv2, padding=1)
                x = torch.relu(x)
                # Adaptive Global Average Pooling
                x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)
                # Fully Connected -> Sigmoid
                logits = torch.nn.functional.linear(x, self.w_fc, self.b_fc)
                prob = torch.sigmoid(logits).item()
                return prob >= threshold
        except Exception as e:
            logger.warning(
                "路由器评估出错 '%s': %s, 默认回退使用基座模式 (Clean)",
                audio_path,
                e,
            )
            return False


class LoRADeltaSwitch:
    """LoRA 偏差矩阵安全加载与就地合并控制器"""

    def __init__(self, base_model, lora_weights_dict: Dict[str, "torch.Tensor"], alpha: int = 16, rank: int = 8):
        self.base_model = base_model
        self.scale = alpha / rank
        self.delta_weights: Dict[str, "torch.Tensor"] = {}
        self._precompute_delta(lora_weights_dict)

    def _precompute_delta(self, lora_weights: Dict[str, "torch.Tensor"]):
        for name, param in self.base_model.named_parameters():
            # 兼容支持多级命名遍历
            if any(proj in name for proj in ["q_proj", "v_proj", "up_proj"]):
                lora_a_key = f"{name}.lora_A.weight"
                lora_b_key = f"{name}.lora_B.weight"

                # 若命名属于 peft 标准命名格式
                if lora_a_key in lora_weights and lora_b_key in lora_weights:
                    a = lora_weights[lora_a_key]
                    b = lora_weights[lora_b_key]

                    # 矩阵乘法计算差值 ΔW = B @ A * scale
                    delta = torch.matmul(b, a) * self.scale

                    if delta.shape != param.shape:
                        raise ValueError(
                            f"权重参数规格冲突! 基座模型命名 [{name}] 形状为 {param.shape}, "
                            f"但微调 LoRA 计算出来的差值形状为 {delta.shape}。"
                        )

                    self.delta_weights[name] = delta.to(device=param.device, dtype=param.dtype)
        logger.info("预编译已完成，共计编译了 %d 组层的 LoRA 差值参数", len(self.delta_weights))

    def merge(self):
        """就地叠加差值"""
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in self.delta_weights:
                    param.data.add_(self.delta_weights[name])

    def unmerge(self):
        """就地扣除差值"""
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in self.delta_weights:
                    param.data.sub_(self.delta_weights[name])


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
            self.router = AudioQualityRouter(self.router_path, self._device)

            # 初始化 LoRA Delta Switch
            logger.info("正在计算并载入 LoRA 鲁棒增强权重差值: %s", self.lora_path)
            lora_weights = load_file(self.lora_path)
            self.lora_switch = LoRADeltaSwitch(
                base_model=self.base_model,
                lora_weights_dict=lora_weights,
                alpha=16,
                rank=8
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
            is_degraded = self.router.predict_is_degraded(audio_path, self.degraded_threshold)
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
            is_degraded = self.router.predict_is_degraded(seg.temp_file, self.degraded_threshold)
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
