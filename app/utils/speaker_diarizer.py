# -*- coding: utf-8 -*-
"""
说话人分离模块
基于 CAM++ 的说话人分离，用于多说话人音频分割
"""

import logging

logger = logging.getLogger(__name__)
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import threading
from typing import Any, List, Mapping, Optional, Sequence, cast
from dataclasses import dataclass

import torch

from ..core.config import settings
from ..core.exceptions import DefaultServerErrorException

# 全局 CAM++ pipeline 缓存（单例）
_global_diarization_pipeline: Any | None = None
_diarization_pipeline_lock = threading.Lock()
_diarization_inference_semaphore = threading.BoundedSemaphore(1)


@dataclass
class SpeakerSegment:
    """说话人分段信息"""

    start_ms: int
    end_ms: int
    speaker_id: str
    audio_data: Optional[np.ndarray] = None
    temp_file: Optional[str] = None

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0


@dataclass
class AsrChunk:
    """ASR 分块：包含一段连续音频及其中的说话人片段信息"""

    start_ms: int
    end_ms: int
    temp_file: str
    speaker_turns: List[SpeakerSegment]  # 该时间段内的说话人片段

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_sec(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0


def _resolve_modelscope_device() -> str:
    """根据配置和硬件自动选择 modelscope pipeline 设备
    """
    from ..core.device import detect_device

    return detect_device(settings.DEVICE)


def _move_pipeline_model_to_device(pipeline_instance: Any, modelscope_device: str) -> None:
    """将 pipeline 的底层模型迁移到目标设备。"""
    if hasattr(pipeline_instance, "device_name"):
        pipeline_instance.device_name = modelscope_device
    model = getattr(pipeline_instance, "model", None)
    if model is not None and hasattr(model, "to"):
        pipeline_instance.model = model.to(modelscope_device)


def _create_modelscope_pipeline(
    *,
    task: Any,
    model: str,
    modelscope_device: str,
    model_revision: Optional[str] = None,
) -> Any:
    """创建 modelscope pipeline，并在需要时把底层模型迁移到目标设备。"""
    from modelscope.pipelines import pipeline

    pipeline_kwargs: dict[str, Any] = {
        "task": task,
        "model": model,
        "device": modelscope_device,
    }
    if model_revision is not None:
        pipeline_kwargs["model_revision"] = model_revision

    pipeline_instance = pipeline(**pipeline_kwargs)
    _move_pipeline_model_to_device(pipeline_instance, modelscope_device)
    return pipeline_instance


def _enable_batched_sv(
    pipeline_instance: Any,
    modelscope_device: str,
    max_batch_size: int = 32,
) -> Any:
    """
    对说话人分离 pipeline 启用 batched SV 推理。

    原始 pipeline 的 forward 方法逐个 segment 调用 sv_pipeline 提取 embedding，
    这里改为将所有 segment 拼成一个 batch 一次性推理，大幅减少 GPU 调用次数。
    同时将子 pipeline（sv / vad / change_locator）绑定到指定 device。

    Args:
        pipeline_instance: CAM++ diarization pipeline 实例
        modelscope_device: 设备名称
        max_batch_size: 最大批处理大小，防止 OOM
    """
    if getattr(pipeline_instance, "_batched_sv_enabled", False):
        return pipeline_instance

    from modelscope.utils.constant import Tasks

    config = getattr(pipeline_instance, "config", None)
    if not isinstance(config, Mapping):
        logger.warning("CAM++ pipeline 缺少可读取的 config，跳过 batched SV 优化")
        return pipeline_instance

    sv_model = config.get("speaker_model")
    vad_model = config.get("vad_model")
    change_locator = config.get("change_locator")

    if isinstance(sv_model, str) and sv_model:
        pipeline_instance.sv_pipeline = _create_modelscope_pipeline(
            task=Tasks.speaker_verification,
            model=sv_model,
            modelscope_device=modelscope_device,
        )

    if isinstance(vad_model, str) and vad_model:
        pipeline_instance.vad_pipeline = _create_modelscope_pipeline(
            task=Tasks.voice_activity_detection,
            model=vad_model,
            modelscope_device=modelscope_device,
            model_revision="v2.0.2",
        )

    if isinstance(change_locator, str) and change_locator:
        pipeline_instance.change_locator_pipeline = _create_modelscope_pipeline(
            task=Tasks.speaker_diarization,
            model=change_locator,
            modelscope_device=modelscope_device,
        )

    def batched_forward(self: Any, segments: Sequence[Sequence[Any]]) -> np.ndarray:
        """批量提取说话人 embedding，替代逐段串行推理"""
        sv_model_instance = getattr(getattr(self, "sv_pipeline", None), "model", None)
        emb_size = int(getattr(sv_model_instance, "emb_size", 192))

        if not segments:
            return np.empty((0, emb_size), dtype=np.float32)

        if sv_model_instance is None:
            raise RuntimeError("CAM++ sv_pipeline.model 未初始化")

        all_embeddings: list[np.ndarray] = []
        total_segments = len(segments)
        start_idx = 0

        while start_idx < total_segments:
            end_idx = min(start_idx + max_batch_size, total_segments)
            batch_segments = segments[start_idx:end_idx]

            batch_items: list[np.ndarray] = []
            for segment in batch_segments:
                if len(segment) < 3:
                    continue
                batch_items.append(np.asarray(segment[2], dtype=np.float32))

            if not batch_items:
                start_idx = end_idx
                continue

            batch = np.stack(batch_items, axis=0)

            with torch.no_grad():
                embeddings = sv_model_instance(
                    cast(Any, torch).as_tensor(batch).to(modelscope_device)
                )

            if isinstance(embeddings, torch.Tensor):
                all_embeddings.append(embeddings.detach().cpu().numpy())
            else:
                all_embeddings.append(np.asarray(embeddings, dtype=np.float32))

            start_idx = end_idx

        if not all_embeddings:
            return np.empty((0, emb_size), dtype=np.float32)

        return (
            np.concatenate(all_embeddings, axis=0)
            if len(all_embeddings) > 1
            else all_embeddings[0]
        )

    import types

    pipeline_instance.forward = types.MethodType(batched_forward, pipeline_instance)
    pipeline_instance._batched_sv_enabled = True

    logger.info(
        f"CAM++ 说话人分离启用 batched SV: device={modelscope_device}, "
        f"sv_device={getattr(getattr(pipeline_instance, 'sv_pipeline', None), 'device_name', 'unknown')}, "
        f"vad_device={getattr(getattr(pipeline_instance, 'vad_pipeline', None), 'device_name', 'unknown')}",
    )
    return pipeline_instance


def get_global_diarization_pipeline() -> Any:
    """获取全局说话人分离 pipeline（懒加载单例）"""
    global _global_diarization_pipeline

    with _diarization_pipeline_lock:
        if _global_diarization_pipeline is None:
            try:
                from modelscope.utils.constant import Tasks
                from ..infrastructure.model_utils import resolve_model_path

                model_id = 'iic/speech_campplus_speaker-diarization_common'
                model_path = resolve_model_path(model_id)
                modelscope_device = _resolve_modelscope_device()

                logger.info(
                    f"正在加载 CAM++ 说话人分离模型: {model_path}, device={modelscope_device}",
                )
                _global_diarization_pipeline = _create_modelscope_pipeline(
                    task=Tasks.speaker_diarization,
                    model=model_path,
                    modelscope_device=modelscope_device,
                )
                _global_diarization_pipeline = _enable_batched_sv(
                    _global_diarization_pipeline, modelscope_device
                )
                logger.info("CAM++ 模型加载成功（已启用 batched SV）")
            except Exception as e:
                logger.error(f"CAM++ 模型加载失败: {e}")
                raise DefaultServerErrorException(f"说话人分离模型加载失败: {str(e)}")

    return _global_diarization_pipeline


class SpeakerDiarizer:
    """基于 CAM++ 的说话人分离器"""

    DEFAULT_MIN_SEGMENT_SEC = 1.0
    DEFAULT_SAMPLE_RATE = 16000
    # 短片段最小输出时长（秒），不足则用静音填充
    MIN_OUTPUT_SEC = 3.0
    LOW_ENERGY_SEARCH_WINDOW_MS = 10000
    LOW_ENERGY_CONTEXT_MS = 160
    LOW_ENERGY_STEP_MS = 20

    def __init__(
        self,
        min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
    ):
        self.min_segment_sec = min_segment_sec
        self.min_segment_ms = int(min_segment_sec * 1000)

    def diarize(
        self, audio_path: str, speaker_num: Optional[int] = None
    ) -> List[SpeakerSegment]:
        """执行说话人分离

        Args:
            audio_path: 音频文件路径
            speaker_num: 已知说话人数量（可选，不传则自动检测）

        Returns:
            原始分段列表（未合并）
        """
        try:
            pipeline = get_global_diarization_pipeline()

            pipeline_kwargs = {"merge_thr": 0.90}
            if speaker_num is not None and speaker_num > 0:
                pipeline_kwargs["oracle_num"] = speaker_num

            logger.info(f"开始说话人分离: {audio_path}, params={pipeline_kwargs}")
            with _diarization_inference_semaphore:
                result = pipeline(audio_path, **pipeline_kwargs)

            # 解析结果: {'text': [[start, end, speaker_id], ...]}
            # pipeline 返回类型不确定，需要安全地获取 'text' 字段
            if isinstance(result, dict):
                raw_output = result.get('text', [])
            else:
                raw_output = getattr(result, 'text', []) or []

            segments = []
            unique_speakers = set()
            for seg in raw_output:
                if isinstance(seg, list) and len(seg) == 3:
                    try:
                        start_ms = int(float(seg[0]) * 1000)
                        end_ms = int(float(seg[1]) * 1000)
                        speaker_id = f"说话人{int(seg[2]) + 1}"
                        unique_speakers.add(speaker_id)
                        segments.append(SpeakerSegment(
                            start_ms=start_ms,
                            end_ms=end_ms,
                            speaker_id=speaker_id,
                        ))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"跳过格式错误的片段: {seg}, 错误: {e}")

            logger.info(
                f"[说话人分离] merge_thr={pipeline_kwargs.get('merge_thr', 'auto')}, 原始片段数: {len(segments)}, "
                f"检测到说话人数: {len(unique_speakers)}, 说话人: {unique_speakers}"
            )
            # 诊断日志：打印前20个原始片段
            for i, seg in enumerate(segments[:20]):
                logger.debug(
                    f"[CAM++原始] #{i}: {seg.start_sec:.2f}-{seg.end_sec:.2f}s "
                    f"({seg.duration_sec:.2f}s) {seg.speaker_id}"
                )
            return segments

        except Exception as e:
            error_msg = str(e).lower()

            # 音频太短时，返回默认的单说话人片段
            if "too short" in error_msg:
                logger.warning(f"音频时长过短，CAM++ 无法处理，返回单说话人片段: {e}")

                # 获取音频时长
                try:
                    audio_duration_ms = int(librosa.get_duration(path=audio_path) * 1000)
                except Exception:
                    # 无法获取时长时使用默认值
                    audio_duration_ms = 5000

                return [
                    SpeakerSegment(
                        start_ms=0,
                        end_ms=audio_duration_ms,
                        speaker_id="说话人1",
                    )
                ]

            # 其他异常正常抛出
            logger.error(f"说话人分离失败: {e}")
            raise DefaultServerErrorException(f"说话人分离失败: {str(e)}")

    def merge_consecutive_segments(
        self, segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        """合并同一说话人的连续片段"""
        if not segments:
            return []

        # 按开始时间排序
        sorted_segments = sorted(segments, key=lambda x: x.start_ms)

        merged = []
        current = SpeakerSegment(
            start_ms=sorted_segments[0].start_ms,
            end_ms=sorted_segments[0].end_ms,
            speaker_id=sorted_segments[0].speaker_id,
        )

        for seg in sorted_segments[1:]:
            if seg.speaker_id == current.speaker_id:
                # 同一说话人，扩展结束时间
                current.end_ms = max(current.end_ms, seg.end_ms)
            else:
                # 不同说话人，保存当前段，开始新段
                logger.debug(
                    f"[合并中断] 说话人切换: {current.speaker_id} → {seg.speaker_id} "
                    f"在 {seg.start_sec:.2f}s，保存片段 {current.start_sec:.2f}-{current.end_sec:.2f}s"
                )
                merged.append(current)
                current = SpeakerSegment(
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    speaker_id=seg.speaker_id,
                )

        # 保存最后一段
        merged.append(current)

        logger.info(f"合并同一说话人连续片段: {len(segments)} → {len(merged)}")
        # 诊断日志：打印合并后的前20个片段
        for i, seg in enumerate(merged[:20]):
            logger.debug(
                f"[合并后] #{i}: {seg.start_sec:.2f}-{seg.end_sec:.2f}s "
                f"({seg.duration_sec:.2f}s) {seg.speaker_id}"
            )
        return merged

    def merge_short_segments(
        self, segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        """智能合并短片段

        策略：
        1. 第一层：<10s的片段向后合并（避免孤立短片段）
        2. 第二层：60s累积合并（合并连续片段）
        """
        if not segments:
            return []

        max_segment_sec = settings.MAX_SEGMENT_SEC

        # 按开始时间排序
        sorted_segments = sorted(segments, key=lambda x: x.start_ms)

        # 第一层：<10s累积向后合并（循环计算直到>=10s或超过60s）
        merged = []
        i = 0
        while i < len(sorted_segments):
            seg = sorted_segments[i]

            # 如果>=10s，直接添加
            if seg.duration_sec >= 10.0:
                merged.append(seg)
                i += 1
                continue

            # <10s，开始累积合并
            current_start_ms = seg.start_ms
            current_end_ms = seg.end_ms
            current_duration_sec = seg.duration_sec
            j = i + 1

            # 累积合并，只要<10s且同说话人且不超过60s
            while j < len(sorted_segments) and current_duration_sec < 10.0:
                next_seg = sorted_segments[j]
                if next_seg.speaker_id != seg.speaker_id:
                    break
                new_duration = (next_seg.end_ms - current_start_ms) / 1000.0
                if new_duration > max_segment_sec:
                    break
                current_end_ms = next_seg.end_ms
                current_duration_sec = new_duration
                j += 1

            # 创建合并后的片段
            merged_seg = SpeakerSegment(
                start_ms=current_start_ms,
                end_ms=current_end_ms,
                speaker_id=seg.speaker_id,
            )
            merged.append(merged_seg)

            if j > i + 1:
                logger.debug(
                    f"[第一层] {seg.speaker_id}: "
                    f"累积合并了 {j - i} 个片段，结果 {merged_seg.duration_sec:.1f}s"
                )
            i = j

        # 第二层：60s累积合并
        final_merged = []
        i = 0
        while i < len(merged):
            seg = merged[i]
            current_start_ms = seg.start_ms
            current_end_ms = seg.end_ms
            j = i + 1

            # 累积合并，只要 <= 60s 且同说话人
            while j < len(merged):
                next_seg = merged[j]
                if next_seg.speaker_id != seg.speaker_id:
                    break
                new_duration = (next_seg.end_ms - current_start_ms) / 1000.0
                if new_duration > max_segment_sec:
                    break
                current_end_ms = next_seg.end_ms
                j += 1

            merged_seg = SpeakerSegment(
                start_ms=current_start_ms,
                end_ms=current_end_ms,
                speaker_id=seg.speaker_id,
            )
            final_merged.append(merged_seg)

            if j > i + 1:
                logger.debug(
                    f"[第二层] {seg.speaker_id}: "
                    f"合并了 {j - i} 个片段"
                )
            i = j

        return final_merged

    def _find_low_energy_boundary_ms(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        lower_ms: int,
        upper_ms: int,
    ) -> int:
        lower_ms = max(0, lower_ms)
        upper_ms = max(lower_ms, upper_ms)
        if sample_rate <= 0 or audio_data.size == 0:
            return upper_ms

        context_samples = max(
            1, int(sample_rate * self.LOW_ENERGY_CONTEXT_MS / 1000)
        )
        candidate_points = list(
            range(lower_ms, upper_ms + 1, self.LOW_ENERGY_STEP_MS)
        )
        if not candidate_points or candidate_points[-1] != upper_ms:
            candidate_points.append(upper_ms)

        best_ms = upper_ms
        best_energy = float("inf")
        audio_length = int(audio_data.shape[0])

        for candidate_ms in candidate_points:
            center_sample = int(candidate_ms * sample_rate / 1000)
            start_sample = max(0, center_sample - context_samples // 2)
            end_sample = min(audio_length, center_sample + context_samples // 2)
            if start_sample >= end_sample:
                continue

            window = audio_data[start_sample:end_sample]
            energy = float(np.mean(np.square(window)))
            if energy <= best_energy:
                best_energy = energy
                best_ms = candidate_ms

        return best_ms

    def split_long_segments(
        self,
        segments: List[SpeakerSegment],
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> List[SpeakerSegment]:
        max_segment_ms = int(settings.MAX_SEGMENT_SEC * 1000)
        if max_segment_ms <= 0:
            return segments

        split_segments: List[SpeakerSegment] = []
        for seg in segments:
            if seg.duration_ms <= max_segment_ms:
                split_segments.append(seg)
                continue

            current_start_ms = seg.start_ms
            while seg.end_ms - current_start_ms > max_segment_ms:
                hard_boundary_ms = current_start_ms + max_segment_ms
                lower_boundary_ms = max(
                    current_start_ms + self.min_segment_ms,
                    hard_boundary_ms - self.LOW_ENERGY_SEARCH_WINDOW_MS,
                )
                boundary_ms = self._find_low_energy_boundary_ms(
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    lower_ms=lower_boundary_ms,
                    upper_ms=hard_boundary_ms,
                )
                if boundary_ms <= current_start_ms:
                    boundary_ms = hard_boundary_ms

                split_segments.append(
                    SpeakerSegment(
                        start_ms=current_start_ms,
                        end_ms=boundary_ms,
                        speaker_id=seg.speaker_id,
                    )
                )
                current_start_ms = boundary_ms

            remaining_ms = seg.end_ms - current_start_ms
            if remaining_ms >= self.min_segment_ms:
                split_segments.append(
                    SpeakerSegment(
                        start_ms=current_start_ms,
                        end_ms=seg.end_ms,
                        speaker_id=seg.speaker_id,
                    )
                )
            elif split_segments:
                split_segments[-1].end_ms = seg.end_ms

        if len(split_segments) != len(segments):
            logger.info(
                f"Split long speaker segments by low energy: {len(segments)} -> {len(split_segments)}, max={settings.MAX_SEGMENT_SEC}s",
            )
        return split_segments

    def split_audio_by_speakers(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        speaker_num: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """完整的说话人分离流程

        流程：
        1. 执行CAM++说话人分离
        2. 智能合并短片段（两层合并策略）
           - 第一层：<10s片段累积合并
           - 第二层：60s累积合并
        3. 提取音频数据，保存临时文件

        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录
            speaker_num: 已知说话人数量（可选，不传则自动检测）

        Returns:
            SpeakerSegment 列表
        """
        try:
            # 1. 执行说话人分离
            raw_segments = self.diarize(audio_path, speaker_num=speaker_num)

            if not raw_segments:
                logger.warning("说话人分离未检测到任何片段")
                return []

            # 2. 智能合并短片段（第一个<10s的同说话人片段向后合并）
            final_segments = self.merge_short_segments(raw_segments)

            # 3. Load audio before low-energy splitting and segment extraction.
            logger.info("加载音频并提取片段...")
            audio_data, sr = librosa.load(audio_path, sr=self.DEFAULT_SAMPLE_RATE)
            sample_rate = int(sr)

            final_segments = self.split_long_segments(
                final_segments,
                audio_data,
                sample_rate,
            )
            logger.info(f"智能合并完成: {len(raw_segments)} → {len(final_segments)} 个片段")

            output_dir = output_dir or settings.TEMP_DIR
            os.makedirs(output_dir, exist_ok=True)

            for idx, seg in enumerate(final_segments):
                start_sample = int(seg.start_ms / 1000 * sample_rate)
                end_sample = int(seg.end_ms / 1000 * sample_rate)

                seg.audio_data = audio_data[start_sample:end_sample]

                # 短片段静音填充：不足 MIN_OUTPUT_SEC 则前后补静音
                # 给 ASR 模型足够的音频长度，不引入额外语音内容
                min_samples = int(self.MIN_OUTPUT_SEC * sample_rate)
                if len(seg.audio_data) < min_samples:
                    shortfall = min_samples - len(seg.audio_data)
                    pad_before = shortfall // 2
                    pad_after = shortfall - pad_before
                    silence_before = np.zeros(pad_before, dtype=np.float32)
                    silence_after = np.zeros(pad_after, dtype=np.float32)
                    padded = np.concatenate([silence_before, seg.audio_data, silence_after])
                    logger.debug(
                        f"[短片段填充] {seg.speaker_id} #{idx}: "
                        f"原始 {seg.duration_sec:.2f}s → "
                        f"填充后 {len(padded)/sample_rate:.2f}s"
                    )
                    output_audio = padded
                else:
                    output_audio = seg.audio_data

                # 保存临时文件
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav",
                    dir=output_dir,
                    prefix=f"{seg.speaker_id}_{idx:03d}_",
                )
                temp_path = temp_file.name
                temp_file.close()

                sf.write(temp_path, output_audio, sample_rate)
                seg.temp_file = temp_path

            # 统计
            unique_speakers = sorted(set(seg.speaker_id for seg in final_segments))
            logger.info(
                f"音频分割完成: {len(final_segments)} 个片段, "
                f"{len(unique_speakers)} 个说话人"
            )
            for spk in unique_speakers:
                spk_segs = [s for s in final_segments if s.speaker_id == spk]
                total_time = sum(s.duration_sec for s in spk_segs)
                logger.info(f"  {spk}: {len(spk_segs)} 片段, {total_time:.2f}s")

            return final_segments

        except Exception as e:
            logger.error(f"说话人分离流程失败: {e}")
            raise DefaultServerErrorException(f"说话人分离失败: {str(e)}")

    @staticmethod
    def cleanup_segments(segments: List[SpeakerSegment]) -> None:
        """清理临时文件"""
        for seg in segments:
            if seg.temp_file and os.path.exists(seg.temp_file):
                try:
                    os.remove(seg.temp_file)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {seg.temp_file}, {e}")

    @staticmethod
    def cleanup_chunks(chunks: List[AsrChunk]) -> None:
        """清理 AsrChunk 临时文件"""
        for chunk in chunks:
            if chunk.temp_file and os.path.exists(chunk.temp_file):
                try:
                    os.remove(chunk.temp_file)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {chunk.temp_file}, {e}")

    def build_asr_chunks(
        self,
        audio_path: str,
        speaker_num: Optional[int] = None,
    ) -> tuple[List[AsrChunk], List[SpeakerSegment]]:
        """按说话人切换点构建 ASR 分块

        将相邻说话人片段累加成 ~MAX_SEGMENT_SEC 的大段，
        只在说话人切换点分段，每段可能包含多个说话人。
        ASR 识别后再按时间戳匹配回说话人标签。

        Args:
            audio_path: 音频文件路径
            speaker_num: 已知说话人数量（可选）

        Returns:
            (chunks, speaker_turns): ASR 分块列表和原始说话人片段列表
        """
        # 1. 执行说话人分离
        raw_segments = self.diarize(audio_path, speaker_num=speaker_num)
        if not raw_segments:
            logger.warning("说话人分离未检测到任何片段")
            return [], []

        # 2. 加载音频（提前加载，后续 diarize/merge/split 复用）
        logger.info("加载音频并构建 ASR 分块...")
        audio_data, sr = librosa.load(audio_path, sr=self.DEFAULT_SAMPLE_RATE)
        sample_rate = int(sr)

        # 3. 合并同一说话人连续片段（避免在说话人内部有微小间隙导致分段过碎）
        merged = self.merge_consecutive_segments(raw_segments)

        # 4. 拆分超过 MAX_SEGMENT_SEC 的单说话人长段（在低能量点切割）
        merged = self.split_long_segments(merged, audio_data, sample_rate)

        # 5. 按说话人切换点累加，每段不超过 MAX_SEGMENT_SEC
        max_ms = int(settings.MAX_SEGMENT_SEC * 1000)
        chunk_groups: List[List[SpeakerSegment]] = []
        current_group: List[SpeakerSegment] = [merged[0]]

        for seg in merged[1:]:
            new_duration = seg.end_ms - current_group[0].start_ms
            if new_duration <= max_ms:
                current_group.append(seg)
            else:
                chunk_groups.append(current_group)
                current_group = [seg]
        if current_group:
            chunk_groups.append(current_group)

        # 6. 为每个 chunk 提取音频、保存临时文件
        output_dir = settings.TEMP_DIR
        os.makedirs(output_dir, exist_ok=True)

        chunks: List[AsrChunk] = []
        try:
            for group_idx, group in enumerate(chunk_groups):
                chunk_start_ms = group[0].start_ms
                chunk_end_ms = group[-1].end_ms
                start_sample = int(chunk_start_ms / 1000 * sample_rate)
                end_sample = int(chunk_end_ms / 1000 * sample_rate)

                chunk_audio = audio_data[start_sample:end_sample]

                # 短 chunk 静音填充：不足 MIN_OUTPUT_SEC 则前后补静音
                min_samples = int(self.MIN_OUTPUT_SEC * sample_rate)
                if len(chunk_audio) < min_samples:
                    shortfall = min_samples - len(chunk_audio)
                    pad_before = shortfall // 2
                    pad_after = shortfall - pad_before
                    chunk_audio = np.concatenate([
                        np.zeros(pad_before, dtype=np.float32),
                        chunk_audio,
                        np.zeros(pad_after, dtype=np.float32),
                    ])

                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav",
                    dir=output_dir,
                    prefix=f"chunk_{group_idx:03d}_",
                )
                temp_path = temp_file.name
                temp_file.close()

                sf.write(temp_path, chunk_audio, sample_rate)
                chunks.append(AsrChunk(
                    start_ms=chunk_start_ms,
                    end_ms=chunk_end_ms,
                    temp_file=temp_path,
                    speaker_turns=group,
                ))
        except Exception:
            # 清理已创建的临时文件，避免泄漏
            for chunk in chunks:
                if chunk.temp_file and os.path.exists(chunk.temp_file):
                    try:
                        os.remove(chunk.temp_file)
                    except Exception:
                        pass
            raise

        # 统计日志
        unique_speakers = sorted(set(s.speaker_id for s in merged))
        logger.info(
            f"ASR 分块完成: {len(merged)} 个说话人片段 → {len(chunks)} 个 ASR 分块, "
            f"{len(unique_speakers)} 个说话人"
        )
        for i, chunk in enumerate(chunks):
            logger.debug(
                f"[ASR分块] #{i}: {chunk.start_sec:.2f}-{chunk.end_sec:.2f}s "
                f"({chunk.duration_sec:.2f}s), {len(chunk.speaker_turns)} 个说话人片段"
            )

        return chunks, merged
