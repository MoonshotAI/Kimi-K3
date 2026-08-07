"""向后兼容垫片：从 kimi_multimodal 重导出。

旧的 `from video_understanding import VideoUnderstanding` 继续可用。
两套客户端已合并，统一走 kimi_multimodal.KimiClient。
"""

from kimi_multimodal import (
    ChatSession,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    KimiClient,
    VALID_REASONING_EFFORT,
    VideoUnderstanding,
)

__all__ = [
    "VideoUnderstanding",
    "KimiClient",
    "ChatSession",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "VALID_REASONING_EFFORT",
    "DEFAULT_REASONING_EFFORT",
]
