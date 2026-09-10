"""
Kimi K3 video understanding module - backwards-compatible entry point

History:
- This file used to implement video understanding on its own (the VideoUnderstanding class),
  against the wrong API base (api.moonshot.ai; the correct one is api.moonshot.cn).
- All functionality now lives in KimiClient in kimi_multimodal.py. This file only re-exports,
  so existing `from video_understanding import VideoUnderstanding` imports keep working.

For new features (multi-turn chat, preserved thinking history), use KimiClient / ChatSession directly:
    from kimi_multimodal import KimiClient, ChatSession, create_client
"""

from kimi_multimodal import KimiClient, ChatSession, create_client, VideoUnderstanding

# Keep the previously public API unchanged
__all__ = ["KimiClient", "VideoUnderstanding", "ChatSession", "create_client"]
