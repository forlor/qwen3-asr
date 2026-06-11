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
    pad_offset_sec: float = 0.0  # 前置静音填充时长（秒），用于修正时间戳偏移

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_sec(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0


def get_global_diarization_pipeline() -> Any:
    """获取全局说话人分离 pipeline（懒加载单例）"""
    global _global_diarization_pipeline

    with _diarization_pipeline_lock:
        if _global_diarization_pipeline is None:
            try:
                from pyannote.audio import Pipeline
                import torch
                from ..core.device import detect_device

                logger.info(
                    f"正在加载 pyannote 说话人分离模型 (speaker-diarization-3.1)...",
                )
                
                hf_token = (settings.HF_TOKEN or "").strip()
                if not hf_token:
                    logger.warning("未配置 HF_TOKEN，如果未在离线缓存中，下载 gated 模型将会失败")

                _global_diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token if hf_token else None
                )
                
                if _global_diarization_pipeline is None:
                    raise RuntimeError("Pipeline.from_pretrained 返回了 None")

                device_str = detect_device(settings.DEVICE)
                _global_diarization_pipeline.to(torch.device(device_str))

                logger.info(f"pyannote 模型加载成功 (device={device_str})")
            except Exception as e:
                logger.error(f"pyannote 模型加载失败: {e}")
                raise DefaultServerErrorException(f"说话人分离模型加载失败: {str(e)}")

    return _global_diarization_pipeline


class SpeakerDiarizer:
    """基于 pyannote 的说话人分离器"""

    DEFAULT_MIN_SEGMENT_SEC = 1.0
    DEFAULT_SAMPLE_RATE = 16000
    # 短片段最小输出时长（秒），不足则用静音填充，从 settings 读取
    MIN_OUTPUT_SEC = settings.DIARIZATION_MIN_OUTPUT_SEC
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

            # 动态调整 pyannote 参数
            params = pipeline.parameters(instantiated=True)
            if speaker_num is not None and speaker_num > 0:
                # 当指定人数时，pyannote 通过 num_speakers 参数控制，不需要改 clustering_threshold
                logger.info(f"开始说话人分离: {audio_path}, num_speakers={speaker_num}")
                with _diarization_inference_semaphore:
                    diarization = pipeline(audio_path, num_speakers=speaker_num)
            else:
                # 使用配置的聚类阈值
                clustering_threshold = settings.DIARIZATION_CLUSTERING_THRESHOLD
                params["clustering"]["threshold"] = clustering_threshold
                pipeline.instantiate(params)
                
                logger.info(f"开始说话人分离: {audio_path}, clustering_threshold={clustering_threshold}")
                with _diarization_inference_semaphore:
                    diarization = pipeline(audio_path)

            segments = []
            unique_speakers = set()
            
            # 解析结果: (turn, track, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                try:
                    start_ms = int(turn.start * 1000)
                    end_ms = int(turn.end * 1000)
                    
                    # pyannote speaker label 是 SPEAKER_00, SPEAKER_01 等
                    # 我们转换为 说话人1, 说话人2
                    if str(speaker).startswith("SPEAKER_"):
                        speaker_idx = int(str(speaker).split("_")[1]) + 1
                        speaker_id = f"说话人{speaker_idx}"
                    else:
                        speaker_id = str(speaker)
                        
                    unique_speakers.add(speaker_id)
                    segments.append(SpeakerSegment(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        speaker_id=speaker_id,
                    ))
                except Exception as e:
                    logger.warning(f"跳过格式错误的片段: {turn}, {speaker}, 错误: {e}")

            logger.info(
                f"[说话人分离] 原始片段数: {len(segments)}, "
                f"检测到说话人数: {len(unique_speakers)}, 说话人: {unique_speakers}"
            )
            # 诊断日志：打印前20个原始片段
            for i, seg in enumerate(segments[:20]):
                logger.debug(
                    f"[pyannote原始] #{i}: {seg.start_sec:.2f}-{seg.end_sec:.2f}s "
                    f"({seg.duration_sec:.2f}s) {seg.speaker_id}"
                )
            return segments

        except Exception as e:
            error_msg = str(e).lower()

            # 音频太短时，返回默认的单说话人片段
            if "too short" in error_msg or "size must be greater than" in error_msg:
                logger.warning(f"音频时长过短，pyannote 无法处理，返回单说话人片段: {e}")

                # 获取音频时长
                try:
                    import librosa
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
                pad_offset_sec = 0.0
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
                    pad_offset_sec = pad_before / sample_rate

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
                    pad_offset_sec=pad_offset_sec,
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
