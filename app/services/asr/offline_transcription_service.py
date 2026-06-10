# -*- coding: utf-8 -*-
"""Shared offline transcription workflow."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
from typing import Any, Dict, Optional

from fastapi import Request

from app.core.config import settings
from app.models.common import SampleRate
from app.services.asr.engines import ASRFullResult
from app.services.asr.model_selection import get_default_offline_model_id
from app.services.asr.runtime import OfflineASRRequest, get_runtime_router
from app.services.audio import get_audio_service


logger = logging.getLogger(__name__)


def _get_audio_quality_router():
    """延迟加载 AudioQualityRouter 单例（仅在启用时初始化）。"""
    from app.services.asr.mega_asr_engine import AudioQualityRouter
    from app.core.device import detect_device

    device = detect_device(settings.DEVICE)
    return AudioQualityRouter(
        model_path=settings.AUDIO_QUALITY_ROUTER_PATH,
        device=device,
        threshold=settings.AUDIO_QUALITY_THRESHOLD,
    )


_audio_quality_router = None
_audio_quality_router_lock = threading.Lock()


def _assess_audio_quality(audio_path: str, task_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """评估音频质量，返回 {"label": ..., "degraded_prob": ...} 或 None。"""
    global _audio_quality_router

    if not settings.AUDIO_QUALITY_ENABLED:
        return None

    try:
        if _audio_quality_router is None:
            with _audio_quality_router_lock:
                if _audio_quality_router is None:
                    _audio_quality_router = _get_audio_quality_router()
        result = _audio_quality_router.predict(audio_path)
        task_prefix = f"[{task_id}] " if task_id else ""
        logger.info(
            "%s音频质量评估完成: label=%s, degraded_prob=%.4f",
            task_prefix,
            result["label"],
            result["degraded_prob"],
        )
        return result
    except Exception as exc:
        logger.warning("音频质量评估不可用，跳过: %s", exc)
        return None


@dataclass(frozen=True)
class PreparedAudio:
    normalized_path: str
    duration: float
    original_path: str
    timestamp_scale: float = 1.0


@dataclass(frozen=True)
class OfflineTranscriptionOptions:
    sample_rate: int = 16000
    hotwords: str = ""
    enable_speaker_diarization: bool = True
    word_timestamps: bool = False
    task_id: Optional[str] = None
    speaker_num: Optional[int] = None


class OfflineTranscriptionService:
    """Prepare audio and run the active offline ASR model."""

    def __init__(self) -> None:
        self._audio_service = get_audio_service()

    async def prepare_from_request(
        self,
        *,
        request: Request,
        audio_address: Optional[str],
        task_id: str,
        sample_rate: int,
    ) -> PreparedAudio:
        audio = await self._audio_service.process_from_request(
            request=request,
            audio_address=audio_address,
            task_id=task_id,
            sample_rate=sample_rate,
        )
        return PreparedAudio(
            normalized_path=audio.normalized_path,
            duration=audio.duration,
            original_path=audio.original_path,
            timestamp_scale=audio.timestamp_scale,
        )

    async def prepare_upload(
        self,
        *,
        audio_data: bytes,
        filename: Optional[str],
        task_id: str,
        sample_rate: int,
    ) -> PreparedAudio:
        audio = await self._audio_service.process_upload_file(
            audio_data=audio_data,
            filename=filename,
            task_id=task_id,
            sample_rate=sample_rate,
        )
        return PreparedAudio(
            normalized_path=audio.normalized_path,
            duration=audio.duration,
            original_path=audio.original_path,
            timestamp_scale=audio.timestamp_scale,
        )

    async def transcribe(
        self,
        prepared_audio: PreparedAudio,
        options: OfflineTranscriptionOptions,
    ) -> ASRFullResult:
        model_id = get_default_offline_model_id()
        result = await get_runtime_router().run_offline(
            OfflineASRRequest(
                model_id=model_id,
                audio_path=prepared_audio.normalized_path,
                hotwords=options.hotwords,
                enable_punctuation=True,
                enable_itn=True,
                sample_rate=options.sample_rate or int(SampleRate.RATE_16000),
                enable_speaker_diarization=options.enable_speaker_diarization,
                word_timestamps=options.word_timestamps,
                timestamp_scale=prepared_audio.timestamp_scale,
                task_id=options.task_id,
                speaker_num=options.speaker_num,
            )
        )

        # 音频质量评估（在推理完成后、清理临时文件前执行）
        quality = _assess_audio_quality(
            prepared_audio.normalized_path,
            task_id=options.task_id,
        )
        if quality is not None:
            result.audio_quality = quality

        return result

    def cleanup(self, prepared_audio: Optional[PreparedAudio]) -> None:
        if prepared_audio is None:
            return
        self._audio_service.cleanup(
            prepared_audio.original_path,
            prepared_audio.normalized_path,
        )


_offline_transcription_service: Optional[OfflineTranscriptionService] = None


def get_offline_transcription_service() -> OfflineTranscriptionService:
    global _offline_transcription_service
    if _offline_transcription_service is None:
        _offline_transcription_service = OfflineTranscriptionService()
    return _offline_transcription_service
