# -*- coding: utf-8 -*-
"""
基于wetext的ITN（逆文本标准化）工具模块
使用wetext库提供高质量的中文ITN处理
"""

import logging

import re

logger = logging.getLogger(__name__)

# wetext导入 - 延迟导入以避免初始化问题
_wetext_normalizer = None


def _get_normalizer():
    """获取wetext标准化器实例（单例模式）"""
    global _wetext_normalizer
    if _wetext_normalizer is None:
        try:
            from wetext import Normalizer
            _wetext_normalizer = Normalizer(lang="zh", operator="itn")
            logger.info("WeText ITN模块初始化成功")
        except ImportError as e:
            logger.error(f"导入wetext失败: {e}")
            raise ImportError("请安装wetext库: pip install wetext")
        except Exception as e:
            logger.error(f"初始化wetext失败: {e}")
            raise
    return _wetext_normalizer


def apply_itn_to_text(text: str) -> str:
    """
    对文本应用逆文本标准化（ITN）
    使用wetext库进行高质量的中文ITN处理

    Args:
        text: 语音识别结果文本

    Returns:
        应用ITN后的文本
    """
    if not text or not text.strip():
        return text

    try:
        normalizer = _get_normalizer()
        result = normalizer.normalize(text)
        logger.debug(f"ITN处理: '{text}' -> '{result}'")
        return result
    except Exception as e:
        logger.warning(f"ITN处理失败: {text}, 错误: {str(e)}")
        return text


def normalize_asr_text(text: str, enable_itn: bool) -> str:
    if not enable_itn:
        return text
    return apply_itn_to_text(text)


def deduplicate_repetition(text: str) -> str:
    """检测并去除文本中连续重复 >= 3 次的短语，只保留一次。

    正常语音中的 1-2 次重复（如"对对对"、"是的是的"）不受影响。
    """
    if not text or len(text) < 6:
        return text

    # 匹配 2 字符以上的片段连续出现 3 次及以上的模式，单字符重复（如"好好好"）不受影响
    pattern = re.compile(r"(.{2,}?)\1{2,}")
    result = pattern.sub(r"\1", text)

    if result != text:
        logger.info(f"重复文本去重: {len(text)} -> {len(result)} 字符")
    return result
