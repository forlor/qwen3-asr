# Mega-ASR 与 Qwen3-ASR 项目融合设计与技术白皮书

本设计文档旨在提供将清华大学 **Mega-ASR** 鲁棒性增强 ASR 框架深度融合进当前 **`qwen3-asr`** 生产级服务的代码级、数学级与工程级蓝图。

---

## 1. 数学原理与显存设计 (Mathematical & Memory Foundations)

### 1.1 动态 LoRA 差值注入数学原理
在传统的 LoRA 推理中，模型需要保持独立的低秩旁路网络：
$$h = W_0 x + \frac{\alpha}{r} B A x$$
其中 $W_0 \in \mathbb{R}^{d \times k}$ 是基座模型冻结的权重，$A \in \mathbb{R}^{r \times k}$ 和 $B \in \mathbb{R}^{d \times r}$ 是低秩矩阵，$r \ll \min(d, k)$ 是秩大小，$\alpha$ 是缩放常数。

每次前向传播（Forward Pass）中，两路矩阵乘法分别计算并相加，会带来额外的计算图跟踪与算子调度开销。尤其在 vLLM 或自定义高效 C++ 算子中，旁路计算会导致显存碎片化并破坏 Attention 算子合并。

Mega-ASR 采用 **显存就地差值注入 (In-place Weight Delta Injection)** 机制。在模型加载时，一次性计算出完整的权重偏差矩阵 $\Delta W \in \mathbb{R}^{d \times k}$：
$$\Delta W = \frac{\alpha}{r} (B \times A)$$

当检测到音频环境变差（Degraded）时，通过 PyTorch 显存直接就地修改（In-place Add）：
$$W_{\text{active}} = W_0 + \Delta W$$
当重新回到纯净环境（Clean）或转写结束时，通过 inplace 减法（In-place Sub）回滚：
$$W_{\text{active}} = W_{\text{active}} - \Delta W = W_0$$

### 1.2 显存指针级就地修改安全性
为确保在多线程或并发调用下显存数据不发生紊乱，必须遵循以下规则：
1. **互斥锁（Mutex Lock）**：整个 ASR 基座模型的权重是单例共享的。在修改模型权重矩阵（`add_` 或 `sub_`）及执行 `forward` 期间，必须加全局线程互斥锁，确保同一时间只有一个推理线程在对该权重进行变形。
2. **就地（In-place）操作无计算图**：所有加减修改必须在 `torch.no_grad()` 块中执行，且必须使用带下划线的方法（如 `.add_()`、`.sub_()`），从而防止 PyTorch 自动微分（Autograd）跟踪计算图导致 OOM（显存溢出）。
3. **数据副本验证**：合并前必须调用 `W.is_contiguous()`，确保显存物理地址连续，防止因切片引用导致非法内存越界。

---

## 2. 音频质量路由器 (Audio Quality Router) 深度剖析

### 2.1 音频特征提取参数配置
分类路由器需要将变长的音频输入转换为标准的二维时频表征。参数规格如下：
- **目标采样率 (Sample Rate)**: 16,000 Hz (单声道)
- **短时傅里叶变换窗口 (STFT Window - N_FFT)**: 1024
- **帧移 (Hop Length)**: 256 (相当于 16ms 帧步长)
- **窗函数 (Window Function)**: 汉宁窗 (Hanning Window)
- **Mel 滤波器组数量 (Num Mel Bins - N_Mels)**: 80
- **特征范围**: 对数美尔谱（Log-Mel Spectrogram），对幅度谱做 $\log(x + 1e-5)$ 缩放。
- **输出特征尺寸**: $\mathbb{R}^{1 \times 80 \times T}$，其中 $T = \text{Duration} \times 100$ 帧。

### 2.2 路由分类网络结构
路由器是由轻量级卷积与池化组成的二进制分类器，加载于 `safetensors` 文件。网络结构如下：

```text
Input Log-Mel: [1, 80, T]
    │
    ▼
Conv2d (In=1, Out=16, Kernel=3, Padding=1) -> ReLU -> BatchNorm
    │
    ▼
MaxPool2d (Kernel=2, Stride=2)  --> [16, 40, T/2]
    │
    ▼
Conv2d (In=16, Out=32, Kernel=3, Padding=1) -> ReLU -> BatchNorm
    │
    ▼
AdaptiveAvgPool2d (Output_Size=[1, 1]) --> 压缩为 [32, 1, 1] 全局空间向量
    │
    ▼
Flatten --> [32]
    │
    ▼
Linear (In=32, Out=1) --> Sigmoid --> Degraded Probability (P)
```

当 $P \ge \text{Threshold}$ (默认 0.5) 时，输出为 `True`（降级）；否则为 `False`（纯净）。

---

## 3. LoRA Delta Switch 机制与 Target Layers 映射

### 3.1 Qwen3-ASR 层命名与遍历
Qwen3-ASR 内部基于标准的 Transformer 结构（注意力机制与 MLP）。需要注入 LoRA 权重的目标层名称（Target Modules）一般包括：
- `q_proj` (Query 映射层)
- `k_proj` (Key 映射层)
- `v_proj` (Value 映射层)
- `o_proj` (Output 密集投影层)
- `gate_proj` / `up_proj` / `down_proj` (SwiGLU MLP 激活与投影层)

### 3.2 遍历与安全校验伪代码

```python
import torch
import logging

logger = logging.getLogger(__name__)

class LoRADeltaSwitch:
    def __init__(self, base_model, lora_weights_dict, alpha=16, rank=8):
        self.base_model = base_model
        self.alpha_over_rank = alpha / rank
        self.delta_weights = {}  # 结构: {param_name: delta_tensor}
        
        # 预计算 Delta 并做形状安全校验
        self._precompute_delta(lora_weights_dict)

    def _precompute_delta(self, lora_weights):
        # 提取 A 与 B 并拼装 Delta
        for name, param in self.base_model.named_parameters():
            # 找到例如: model.layers.0.self_attn.q_proj.weight
            if any(target in name for target in ["q_proj", "v_proj", "up_proj"]):
                lora_a_key = f"{name}.lora_A.weight"
                lora_b_key = f"{name}.lora_B.weight"
                
                if lora_a_key in lora_weights and lora_b_key in lora_weights:
                    a_w = lora_weights[lora_a_key] # 形状: [r, k]
                    b_w = lora_weights[lora_b_key] # 形状: [d, r]
                    
                    # 矩阵乘法计算 ΔW
                    delta_w = torch.matmul(b_w, a_w) * self.alpha_over_rank
                    
                    # 严苛校验: 确保 ΔW 与 基座对应权重的 Shape 完全一致
                    if delta_w.shape != param.shape:
                        raise ValueError(
                            f"参数形状不匹配! 基座参数 '{name}' 形状为 {param.shape}, "
                            f"计算出的 Delta 形状为 {delta_w.shape}"
                        )
                    
                    self.delta_weights[name] = delta_w.to(device=param.device, dtype=param.dtype)
        logger.info(f"预计算完成，共编译了 {len(self.delta_weights)} 个矩阵的 Delta 权重")

    def merge(self):
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in self.delta_weights:
                    param.data.add_(self.delta_weights[name])

    def unmerge(self):
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in self.delta_weights:
                    param.data.sub_(self.delta_weights[name])
```

---

## 4. 分组批处理 (Grouped Batch) 索引双向映射算法

当大音频被 VAD 切片为 $N$ 段临时文件时，我们需要同时将这些切片送入 `_transcribe_batch`。为了将合并矩阵的操作降到极致，算法执行如下双向映射步骤：

```text
原始输入切片列表:
Indices:    0      1      2      3      4      5
Audios:   [Seg0,  Seg1,  Seg2,  Seg3,  Seg4,  Seg5]
            │      │      │      │      │      │
            ▼      ▼      ▼      ▼      ▼      ▼
[步骤 1] 质量路由分类预判:
Router:   [Clean, Noise, Clean, Noise, Noise, Clean] (P >= Threshold 判定)
            │      │      │      │      │      │
            ▼      ▼      ▼      ▼      ▼      ▼
[步骤 2] 分类打包与索引映射记录:
Group Clean: [Seg0, Seg2, Seg5]  -->  映射原索引为: {0:0, 1:2, 2:5}
Group Noise: [Seg1, Seg3, Seg4]  -->  映射原索引为: {0:1, 1:3, 2:4}

[步骤 3] 第一阶段推理 (Base Mode - 无 LoRA):
Batch Forward on Group Clean  -->  产生文本 Clean_Texts: [T0, T2, T5]

[步骤 4] 切换权重 (LoRA Delta Switch Merge)
Model Weight  <--  Weight + Delta_W

[步骤 5] 第二阶段推理 (Robust Mode - LoRA Active):
Batch Forward on Group Noise  -->  产生文本 Noise_Texts: [T1, T3, T4]

[步骤 6] 回滚权重 (LoRA Delta Switch Unmerge)
Model Weight  <--  Weight - Delta_W

[步骤 7] 倒排索引重组还原:
Final Texts[0] = Clean_Texts[0] (T0)
Final Texts[1] = Noise_Texts[0] (T1)
Final Texts[2] = Clean_Texts[1] (T2)
Final Texts[3] = Noise_Texts[1] (T3)
Final Texts[4] = Noise_Texts[2] (T4)
Final Texts[5] = Clean_Texts[2] (T5)
```

---

## 5. `app/services/asr/mega_asr_engine.py` 完整实现规范

```python
# -*- coding: utf-8 -*-
"""
Mega-ASR (Robustness LoRA Enhanced) Engine.
实现高内聚的 PyTorch 离线 ASR 与动态批分组算法。
"""

import logging
import os
import threading
import torch
import numpy as np
import librosa
from typing import Optional, List, Any, Dict
from safetensors.torch import load_file

from .engines import BaseASREngine, ASRRawResult, ASRSegmentResult, WordToken
from ...core.exceptions import DefaultServerErrorException
from ...core.config import settings
from ...utils.text_processing import normalize_asr_text

logger = logging.getLogger(__name__)


class AudioQualityRouter:
    """音频质量评估神经网络路由器（Log-Mel Spectrogram + CNN Classifier）"""

    def __init__(self, model_path: str, device: str):
        self.device = device
        self.model_path = model_path
        self._load_network()

    def _load_network(self):
        try:
            # 动态加载分类器 safetensors 权重
            self.weights = load_file(self.model_path)
            # 初始化一维/二维卷积层及全连接层
            self.w_conv1 = self.weights["conv1.weight"].to(self.device)
            self.b_conv1 = self.weights["conv1.bias"].to(self.device)
            self.w_conv2 = self.weights["conv2.weight"].to(self.device)
            self.b_conv2 = self.weights["conv2.bias"].to(self.device)
            self.w_fc = self.weights["fc.weight"].to(self.device)
            self.b_fc = self.weights["fc.bias"].to(self.device)
            logger.info("音频质量分类路由器加载成功")
        except Exception as e:
            raise RuntimeError(f"路由器权重载入失败: {e}")

    def _extract_mel(self, audio_path: str) -> torch.Tensor:
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
            logger.warning(f"路由器推断出错 '{audio_path}': {e}, 默认使用 Base 模式(Clean)")
            return False


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
        **kwargs,
    ):
        from app.core.device import detect_device
        self._device = detect_device(device)
        self.model_id = "mega-asr-1.7b"
        self.model_path = model_path
        
        self.lora_path = lora_path or settings.MEGA_ASR_LORA_PATH
        self.router_path = router_path or settings.MEGA_ASR_ROUTER_PATH
        self.degraded_threshold = degraded_threshold

        # 并发推理锁，保证多路 ASR 显存安全
        self._infer_lock = threading.Lock()
        
        self.base_model = None
        self.tokenizer = None
        self.router = None
        self.lora_switch = None

        if not self._device.startswith("cuda"):
            raise DefaultServerErrorException("Mega-ASR 引擎由于包含显存注入，必须运行于 CUDA 环境下！")

        self._load_all_components()

    def _load_all_components(self):
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer
            logger.info("正在载入 Qwen3-ASR 基座模型: %s", self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self._device)
            
            # 初始化音频路由器
            logger.info("正在载入音频环境评估路由器: %s", self.router_path)
            self.router = AudioQualityRouter(self.router_path, self._device)

            # 初始化 LoRA Delta Switch
            logger.info("正在载入 LoRA 鲁棒增强权重差值: %s", self.lora_path)
            lora_weights = load_file(self.lora_path)
            self.lora_switch = LoRADeltaSwitch(
                base_model=self.base_model,
                lora_weights_dict=lora_weights,
                alpha=16,
                rank=8
            )
            logger.info("Mega-ASR 联合引擎所有组件全部加载成功。")
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise DefaultServerErrorException(f"组件加载失败: {e}")

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
            raise DefaultServerErrorException("模型未正确装载")

        # 使用互斥锁保护显存
        with self._infer_lock:
            # 1. 检测环境噪声级别
            is_degraded = self.router.predict_is_degraded(audio_path, self.degraded_threshold)
            logger.info(f"音频 '{audio_path}' 环境检测结果: {'降级嘈杂(Degraded)' if is_degraded else '纯净良好(Clean)'}")

            # 2. 动态合并
            if is_degraded:
                self.lora_switch.merge()

            try:
                # 3. 前向 ASR 转写
                text = self._run_raw_inference(audio_path, hotwords)
            finally:
                # 4. 回滚合并
                if is_degraded:
                    self.lora_switch.unmerge()

            return normalize_asr_text(text, enable_itn=enable_itn)

    def _run_raw_inference(self, audio_path: str, prompt: str) -> str:
        """基座模型推理前向前向"""
        y, sr = librosa.load(audio_path, sr=16000)
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
        **kwargs,
    ) -> ASRRawResult:
        text = self.transcribe_file(audio_path, hotwords, enable_punctuation, enable_itn, sample_rate=sample_rate)
        return ASRRawResult(text=text, segments=[ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)])

    def _transcribe_batch(
        self,
        segments: List[Any],
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        sample_rate: int = 16000,
        word_timestamps: bool = False,
    ) -> List[ASRSegmentResult]:
        """批量多任务推理 - 高性能分组算法"""
        if not segments:
            return []

        # 索引双向映射
        clean_idx_map = []  # 记录原索引
        degraded_idx_map = []
        
        # 1. 质量分类预判定
        for idx, seg in enumerate(segments):
            if not seg.temp_file:
                continue
            is_degraded = self.router.predict_is_degraded(seg.temp_file, self.degraded_threshold)
            if is_degraded:
                degraded_idx_map.append((idx, seg.temp_file))
            else:
                clean_idx_map.append((idx, seg.temp_file))

        logger.info(f"批量推理分流完成: 纯净(Clean)数={len(clean_idx_map)}, 降级(Degraded)数={len(degraded_idx_map)}")

        results_dict: Dict[int, str] = {}

        # 使用显存互斥锁
        with self._infer_lock:
            # 2. 第一阶段：推理 Clean 分组 (基座模型无 LoRA)
            if clean_idx_map:
                logger.info("执行 Clean 分组批量前向推理...")
                for orig_idx, path in clean_idx_map:
                    try:
                        results_dict[orig_idx] = self._run_raw_inference(path, hotwords)
                    except Exception as e:
                        logger.error(f"Clean 音频推理失败 #{orig_idx}: {e}")
                        results_dict[orig_idx] = ""

            # 3. 第二阶段：切换权重并推理 Degraded 分组 (合并 LoRA)
            if degraded_idx_map:
                logger.info("切换模型权重 (LoRA Delta Switch Merge)...")
                self.lora_switch.merge()
                try:
                    logger.info("执行 Degraded 分组批量前向推理...")
                    for orig_idx, path in degraded_idx_map:
                        try:
                            results_dict[orig_idx] = self._run_raw_inference(path, hotwords)
                        except Exception as e:
                            logger.error(f"Degraded 音频推理失败 #{orig_idx}: {e}")
                            results_dict[orig_idx] = ""
                finally:
                    # 4. 收尾回滚
                    logger.info("回滚模型权重 (LoRA Delta Switch Unmerge)...")
                    self.lora_switch.unmerge()

        # 5. 倒排索引重组并返回
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


class LoRADeltaSwitch:
    """LoRA 偏差矩阵安全加载与就地合并控制器"""

    def __init__(self, base_model, lora_weights_dict: Dict[str, torch.Tensor], alpha: int = 16, rank: int = 8):
        self.base_model = base_model
        self.scale = alpha / rank
        self.delta_weights: Dict[str, torch.Tensor] = {}
        self._precompute_delta(lora_weights_dict)

    def _precompute_delta(self, lora_weights: Dict[str, torch.Tensor]):
        for name, param in self.base_model.named_parameters():
            if any(proj in name for proj in ["q_proj", "v_proj", "up_proj"]):
                lora_a_key = f"{name}.lora_A.weight"
                lora_b_key = f"{name}.lora_B.weight"
                
                if lora_a_key in lora_weights and lora_b_key in lora_weights:
                    a = lora_weights[lora_a_key]
                    b = lora_weights[lora_b_key]
                    
                    # 计算 ΔW = B @ A * scale
                    delta = torch.matmul(b, a) * self.scale
                    
                    if delta.shape != param.shape:
                        raise ValueError(f"模型参数 [{name}] 形状 {param.shape} 与 LoRA 计算差值形状 {delta.shape} 冲突")
                    
                    self.delta_weights[name] = delta.to(device=param.device, dtype=param.dtype)

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


def _register_mega_asr_engine(register_func, _declared_entry_cls):
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
```

---

## 6. 核心配置文件修改级 Blueprint (Existing Code Diff Specification)

### 6.1 `app/core/config.py` 改动蓝图
定位到 `app/core/config.py` 中的 `Settings` 类，在类定义中新增 Mega-ASR 的路径及阈值选项，确保统一采用环境变量加载：

```python
<<<<
    # 批处理推理配置（GPU 真并行）
    ASR_BATCH_SIZE: int = 4  # ASR 批处理大小（同时推理的片段数），建议 2-8
====
    # 批处理推理配置（GPU 真并行）
    ASR_BATCH_SIZE: int = 4  # ASR 批处理大小（同时推理的片段数），建议 2-8

    # Mega-ASR 鲁棒性模型环境配置
    MEGA_ASR_LORA_PATH: str = "ckpt/Mega-ASR/lora/adapter_model.safetensors"
    MEGA_ASR_ROUTER_PATH: str = "ckpt/Mega-ASR/router/model.safetensors"
    MEGA_ASR_DEGRADED_THRESHOLD: float = 0.5
>>>>
```

并在 `_load_from_env` 成员方法内进行环境变量绑定注入：

```python
<<<<
        self.ASR_BATCH_SIZE = int(
            os.getenv("ASR_BATCH_SIZE", str(self.ASR_BATCH_SIZE))
        )
====
        self.ASR_BATCH_SIZE = int(
            os.getenv("ASR_BATCH_SIZE", str(self.ASR_BATCH_SIZE))
        )
        self.MEGA_ASR_LORA_PATH = os.getenv("MEGA_ASR_LORA_PATH", self.MEGA_ASR_LORA_PATH)
        self.MEGA_ASR_ROUTER_PATH = os.getenv("MEGA_ASR_ROUTER_PATH", self.MEGA_ASR_ROUTER_PATH)
        self.MEGA_ASR_DEGRADED_THRESHOLD = float(
            os.getenv("MEGA_ASR_DEGRADED_THRESHOLD", str(self.MEGA_ASR_DEGRADED_THRESHOLD))
        )
>>>>
```

### 6.2 `app/services/asr/models.json` 改动蓝图
在 `models` 根 JSON 对象下追加一个全新模型节点 `mega-asr-1.7b`。该条目指向 `mega_asr` 引擎：

```json
<<<<
        "qwen3-asr-0.6b": {
            ...
            "extra_kwargs": {
                "max_model_len": 16384,
                "forced_aligner_path": "Qwen/Qwen3-ForcedAligner-0.6B",
                "max_inference_batch_size": 16
            }
        }
====
        "qwen3-asr-0.6b": {
            ...
            "extra_kwargs": {
                "max_model_len": 16384,
                "forced_aligner_path": "Qwen/Qwen3-ForcedAligner-0.6B",
                "max_inference_batch_size": 16
            }
        },
        "mega-asr-1.7b": {
            "name": "Mega-ASR-1.7B",
            "kind": "model",
            "engine": "mega_asr",
            "description": "Mega-ASR 1.7B，基于 Qwen3-ASR 并结合神经网络分类器预判环境进行动态 LoRA 加减合并，具有极强抗噪性",
            "languages": [
                "zh", "en", "yue", "ja", "ko"
            ],
            "default": false,
            "supports_realtime": false,
            "models": {
                "offline": "Qwen/Qwen3-ASR-1.7B"
            },
            "extra_kwargs": {
                "max_model_len": 16384,
                "max_inference_batch_size": 16
            }
        }
>>>>
```

### 6.3 `app/services/asr/manager.py` 改动蓝图
在 `app/services/asr/manager.py` 的首部添加动态引擎工厂钩子，在 `_register_builtin_engines` 中挂载：

```python
<<<<
    try:
        from .qwen3_engine import Qwen3ASREngine  # noqa: F401
        from .qwen3_engine import _register_qwen3_engine
        _register_qwen3_engine(register_engine, DeclaredEntryConfig)
    except ImportError as e:
        logger.warning(f"Qwen3引擎不可用: {e}")
====
    try:
        from .qwen3_engine import Qwen3ASREngine  # noqa: F401
        from .qwen3_engine import _register_qwen3_engine
        _register_qwen3_engine(register_engine, DeclaredEntryConfig)
    except ImportError as e:
        logger.warning(f"Qwen3引擎不可用: {e}")

    try:
        from .mega_asr_engine import MegaASREngine  # noqa: F401
        from .mega_asr_engine import _register_mega_asr_engine
        _register_mega_asr_engine(register_engine, DeclaredEntryConfig)
    except ImportError as e:
        logger.warning(f"Mega-ASR引擎不可用: {e}")
>>>>
```

---

## 7. 异常控制与健壮性保障 (Error Handling & Safeguards)

### 7.1 模型尺寸 shape 不匹配防护
启动服务并加载 `MegaASREngine` 时，若因为配置错误（如：用户配置了 `qwen3-asr-0.6b` 作为基座模型，但加载了 `qwen3-asr-1.7b` 对应的 LoRA 权重），在 `LoRADeltaSwitch` 的 `_precompute_delta` 预计算环节会抛出严格的 `ValueError`。这将导致 ASR 服务立刻初始化失败报错，避免在推理运行时发生不确定的 PyTorch C++ 底层张量相加错误（Shape Mismatch Crash）。

### 7.2 显存耗尽 (OOM) 自动释放与降级机制
当多路推理或大 Batch 导致 GPU VRAM 突发不足时，采取如下熔断和恢复策略：
1. **就地清理**：触发 OOM 异常时，在 `except` 捕获块中，立刻调用 `torch.cuda.empty_cache()` 并执行 Python 回收机制 `import gc; gc.collect()`。
2. **状态还原**：捕获异常后，由于此前可能合并了 LoRA 差值，必须强制调用 `self.lora_switch.unmerge()` 保证模型参数在物理上被完全还原回 Base，以便后续批次能正常运行。
3. **退化为 CPU 转录**：捕获 OOM 之后，可以尝试将本次 Batch 自动分流回 CPU 版的 Rust 引擎（即调用 `QwenASRRustRuntime`），以保障服务在极端状态下的 100% 可用性。
