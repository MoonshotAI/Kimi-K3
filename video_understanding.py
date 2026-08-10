"""
Kimi K3 视频理解模块 —— 向后兼容入口

历史说明：
- 早期本文件独立实现了视频理解（VideoUnderstanding 类），且使用了错误的 API 基址
  api.moonshot.ai（正确应为 api.moonshot.cn）。
- 现所有能力已统一到 kimi_multimodal.py 的 KimiClient，本文件仅做重导出，
  避免破坏既有 import `from video_understanding import VideoUnderstanding`。

如需新功能（多轮对话、thinking history 保留等），请直接使用 KimiClient / ChatSession：
    from kimi_multimodal import KimiClient, ChatSession, create_client
"""

from kimi_multimodal import KimiClient, ChatSession, create_client, VideoUnderstanding

# 保持旧有公开的 API 不变
__all__ = ["KimiClient", "VideoUnderstanding", "ChatSession", "create_client"]
