"""Kimi K3 多模态客户端与会话管理。

统一 API base 为 https://api.moonshot.cn/v1，reasoning_effort 对齐 README
取值 low/high/max（默认 max）。ChatSession 自动把 API 返回的 reasoning_content
与 tool_calls 原样存回 messages，满足 K3 多轮对话对 thinking history 的要求。
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

import openai

DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "kimi-k3"
VALID_REASONING_EFFORT = ("low", "high", "max")
DEFAULT_REASONING_EFFORT = "max"

__all__ = [
    "KimiClient",
    "ChatSession",
    "VideoUnderstanding",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "VALID_REASONING_EFFORT",
    "DEFAULT_REASONING_EFFORT",
]


class KimiClient:
    """Kimi K3 客户端：纯文本对话入口 + 多模态发送 + 流式（reasoning/content 分离）。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> None:
        self.api_key = api_key or os.environ.get("MOONSHOT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = model
        self.reasoning_effort = self._validate_effort(reasoning_effort)
        # 让无 key 也能构造对象（便于离线 mock 测试）；真正调用时再报错
        self.client = openai.OpenAI(api_key=self.api_key or "missing", base_url=self.base_url)

    @staticmethod
    def _validate_effort(effort: str) -> str:
        if effort not in VALID_REASONING_EFFORT:
            raise ValueError(
                f"reasoning_effort 必须是 {VALID_REASONING_EFFORT} 之一，收到: {effort!r}"
            )
        return effort

    # ---- 纯文本对话入口 ----
    def chat(self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs: Any):
        """纯文本对话入口（messages 内可携带多模态 content）。"""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            reasoning_effort=self.reasoning_effort,
            **kwargs,
        )

    # ---- 流式：reasoning 与 content 分开产出 ----
    def stream_chat(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Iterator[Dict[str, str]]:
        """流式输出，逐 token 产出 {"type": "reasoning"|"content", "delta": str}。"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            reasoning_effort=self.reasoning_effort,
            **kwargs,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                yield {"type": "reasoning", "delta": rc}
            if getattr(delta, "content", None):
                yield {"type": "content", "delta": delta.content}

    # ---- 多模态发送 ----
    def send_image(
        self,
        image: str,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        url = self._to_image_url(image)
        content = [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": prompt},
        ]
        msg = {"role": "user", "content": content}
        if messages is None:
            messages = []
        messages.append(msg)
        return self.chat(messages, **kwargs)

    def send_video(
        self,
        video: str,
        prompt: str,
        frames: int = 8,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        image_urls = self._extract_video_frames(video, frames)
        content: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": u}} for u in image_urls
        ]
        content.append({"type": "text", "text": prompt})
        msg = {"role": "user", "content": content}
        if messages is None:
            messages = []
        messages.append(msg)
        return self.chat(messages, **kwargs)

    # ---- 工具方法 ----
    @staticmethod
    def _to_image_url(image: str) -> str:
        """路径 -> base64 data URL；URL/data URL 原样返回。"""
        if isinstance(image, str) and os.path.exists(image):
            ext = os.path.splitext(image)[1].lower().lstrip(".")
            mime = {
                "jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                "gif": "gif", "webp": "webp",
            }.get(ext, "jpeg")
            with open(image, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/{mime};base64,{b64}"
        return image

    @staticmethod
    def _extract_video_frames(video_path: str, num_frames: int) -> List[str]:
        """从视频中均匀抽取 num_frames 帧，返回 base64 JPEG data URL 列表。"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        try:
            import imageio.v3 as iio
            import io
            import numpy as np
            from PIL import Image
        except ImportError as e:  # pragma: no cover - 依赖缺失分支
            raise ImportError(
                "视频抽帧需要依赖：pip install imageio[ffmpeg] pillow numpy"
            ) from e

        meta = iio.immeta(video_path)
        total = meta.get("n_frames") or meta.get("duration_frames")
        if not total:
            total = sum(1 for _ in iio.imiter(video_path))
        indices = np.linspace(0, max(total - 1, 0), num=max(num_frames, 1), dtype=int)

        urls: List[str] = []
        for idx in indices:
            frame = iio.imread(video_path, index=int(idx))
            buf = io.BytesIO()
            Image.fromarray(frame).convert("RGB").save(buf, format="JPEG", quality=85)
            urls.append(f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}")
        return urls


class ChatSession:
    """多轮对话会话：自动保留完整 thinking history（reasoning_content + tool_calls）。"""

    def __init__(
        self,
        client: KimiClient,
        system: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.client = client
        if reasoning_effort is not None:
            client.reasoning_effort = KimiClient._validate_effort(reasoning_effort)
        self.system: Optional[str] = system
        self.messages: List[Dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        """清空历史，但保留 system 消息。"""
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def ask(self, user_text: str, **kwargs: Any):
        """非流式提问，自动把完整 assistant 消息（含 thinking）存回 messages。"""
        self.messages.append({"role": "user", "content": user_text})
        resp = self.client.chat(self.messages, **kwargs)
        self._store_assistant(resp)
        return resp

    def stream_ask(
        self, user_text: str, **kwargs: Any
    ) -> Iterator[Dict[str, str]]:
        """流式提问，先 yield 各 token 事件，结束时把完整 assistant 消息存回 messages。"""
        self.messages.append({"role": "user", "content": user_text})
        reasoning_parts: List[str] = []
        content_parts: List[str] = []
        tool_calls: Any = None
        for ev in self.client.stream_chat(self.messages, **kwargs):
            if ev["type"] == "reasoning":
                reasoning_parts.append(ev["delta"])
            else:
                content_parts.append(ev["delta"])
            yield ev
        # 流式里若带 tool_calls 需另外抓取；这里保守置空，避免覆盖
        assistant: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts),
            "reasoning_content": "".join(reasoning_parts),
        }
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        self.messages.append(assistant)

    def add_tool_result(self, tool_call_id: str, result: str) -> None:
        """把工具执行结果作为 tool 角色消息追加进历史。"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        })

    def _store_assistant(self, resp) -> None:
        """从非流式响应里取出完整 assistant 消息原样存回。"""
        msg = resp.choices[0].message
        assistant: Dict[str, Any] = {
            "role": "assistant",
            "content": getattr(msg, "content", None),
            "reasoning_content": getattr(msg, "reasoning_content", None),
        }
        tc = getattr(msg, "tool_calls", None)
        if tc:
            assistant["tool_calls"] = tc
        self.messages.append(assistant)


class VideoUnderstanding(KimiClient):
    """面向视频理解的便捷子类，等价于 KimiClient 但默认聚焦视频任务。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        default_frames: int = 8,
    ) -> None:
        super().__init__(api_key, base_url, model, reasoning_effort)
        self.default_frames = default_frames

    def understand(
        self,
        video: str,
        prompt: str,
        frames: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        return self.send_video(
            video, prompt, frames=frames or self.default_frames,
            messages=messages, **kwargs,
        )
