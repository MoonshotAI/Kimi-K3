"""
Kimi K3 多模态 AI 客户端
提供文本对话、图片识别/生成、视频理解/生成、图生视频等功能。

说明：
- 原 video_understanding.py 的视频功能已合并到本文件的 KimiClient，
  video_understanding.py 现在只做向后兼容重导出（见文件底部）。
- API base 统一为官方 OpenAI 兼容端点 https://api.moonshot.cn/v1
  （旧版 video_understanding.py 误用了 api.moonshot.ai，已修正）。
- reasoning_effort 对齐官方 README，仅支持 "low" / "high" / "max"（默认 "max"）。
"""

import base64
import os
import tempfile
from pathlib import Path
from typing import Optional, Generator, List, Tuple, Dict, Any, Union

import httpx
import requests
from openai import OpenAI


# 统一 API base：Kimi / Moonshot 官方 OpenAI 兼容端点。
# 原 video_understanding.py 误用了 api.moonshot.ai，此处统一修正为 .cn。
API_BASE = "https://api.moonshot.cn/v1"

# 对齐 README：reasoning_effort 仅支持以下取值，默认 "max"。
VALID_REASONING_EFFORTS = ("low", "high", "max")
DEFAULT_REASONING_EFFORT = "max"


class KimiClient:
    """Kimi K3 多模态 AI 客户端（统一入口）。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "kimi-k3",
        api_base: str = API_BASE,
        bypass_proxy: bool = True,
    ):
        """
        初始化 Kimi 客户端

        Args:
            api_key: Kimi API 密钥，未提供则从环境变量 KIMI_API_KEY 读取
            model: 模型名称，默认 kimi-k3
            api_base: API 基址，默认官方 OpenAI 兼容端点
            bypass_proxy: 是否绕过系统代理直连（默认 True，国内直连更快）
        """
        self.api_key = api_key or os.environ.get("KIMI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "请提供 API 密钥或设置环境变量 KIMI_API_KEY\n"
                "  export KIMI_API_KEY=\"your_api_key\"\n"
                "  API key 从 https://platform.kimi.ai 获取"
            )

        self.model = model
        self.api_base = api_base

        # 绕过代理，直接连接 API（避免本地代理环境变量干扰）
        http_client = httpx.Client(trust_env=not bypass_proxy)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=http_client,
        )

    # ============================================================
    # 文本对话
    # ============================================================

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Union[str, Generator[Tuple[str, str], None, None]]:
        """
        纯文本对话（一次性）。

        Args:
            prompt: 用户输入
            system: 可选的 system 提示词
            stream: 是否流式输出。流式时返回生成器，按 (kind, text) 分块，
                    kind 为 "reasoning"（思考过程）或 "content"（最终回答）
            reasoning_effort: "low" / "high" / "max"

        Returns:
            非流式返回完整文本；流式返回 (kind, text) 生成器
        """
        session = ChatSession(self, system=system, reasoning_effort=reasoning_effort)
        return session.chat(prompt, stream=stream)

    def create_session(
        self,
        system: Optional[str] = None,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        model: Optional[str] = None,
    ) -> "ChatSession":
        """创建一个多轮会话（自动保留 thinking history）。"""
        return ChatSession(self, system=system, reasoning_effort=reasoning_effort, model=model)

    # ============================================================
    # 图片识别功能
    # ============================================================

    def analyze_image(
        self,
        image_source: str,
        prompt: str = "请详细描述这张图片的内容。",
        stream: bool = False,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Union[Generator[str, None, None], str]:
        """
        分析图片内容

        Args:
            image_source: 图片源，支持 本地路径 / http(s) URL / base64: 前缀
            prompt: 分析提示词
            stream: 是否流式输出
            reasoning_effort: "low" / "high" / "max"

        Returns:
            stream=True 返回生成器；否则返回完整响应字符串
        """
        image_b64, mime_type = self._load_image(image_source)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            reasoning_effort=reasoning_effort,
        )

        if stream:
            return self._process_stream(response)
        else:
            return response.choices[0].message.content

    def describe_image(self, image_source: str) -> str:
        """快速描述图片内容"""
        return self.analyze_image(
            image_source,
            prompt="请用2-3句话简洁地描述这张图片的主要内容。",
            stream=False,
        )

    def identify_objects(self, image_source: str) -> str:
        """识别图片中的物体"""
        return self.analyze_image(
            image_source,
            prompt="请列出图片中所有可识别的物体，并简要描述它们的位置和特征。",
            stream=False,
        )

    def analyze_scene(self, image_source: str) -> str:
        """分析图片场景"""
        return self.analyze_image(
            image_source,
            prompt="请分析这张图片的场景，包括：环境、氛围、光线、色调、可能的时间和地点。",
            stream=False,
        )

    def extract_text_from_image(self, image_source: str) -> str:
        """从图片中提取文字（OCR）"""
        return self.analyze_image(
            image_source,
            prompt="请识别并提取图片中的所有文字内容，保持原始格式。",
            stream=False,
        )

    def answer_image_question(self, image_source: str, question: str) -> str:
        """基于图片回答特定问题"""
        return self.analyze_image(
            image_source,
            prompt=f"请观察图片后回答以下问题：{question}",
            stream=False,
        )

    # ============================================================
    # 图片生成功能
    # ============================================================

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        style: Optional[str] = None,
    ) -> List[str]:
        """
        生成图片

        Args:
            prompt: 图片生成提示词
            size: 图片尺寸，可选 "1024x1024", "1792x1024", "1024x1792"
            n: 生成图片数量
            style: 风格描述（可选）

        Returns:
            生成的图片 URL 列表
        """
        full_prompt = prompt
        if style:
            full_prompt = f"{prompt}, 风格: {style}"

        response = self.client.images.generate(
            model="kimi-k3-image",
            prompt=full_prompt,
            size=size,
            n=n,
        )

        return [item.url for item in response.data]

    def generate_and_save_image(
        self,
        prompt: str,
        output_path: str,
        size: str = "1024x1024",
        style: Optional[str] = None,
    ) -> str:
        """
        生成图片并保存到本地

        Returns:
            保存的文件路径
        """
        urls = self.generate_image(prompt, size=size, style=style)

        if not urls:
            raise RuntimeError("图片生成失败")

        url = urls[0]
        print("正在下载生成的图片...")

        resp = requests.get(url, stream=True, timeout=120, proxies={})
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"图片已保存: {output_path}")
        return output_path

    # ============================================================
    # 视频理解功能
    # ============================================================

    def analyze_video(
        self,
        video_source: str,
        prompt: str = "请仔细观看这个视频，详细描述视频中的内容：画面里有什么？发生了什么？",
        stream: bool = True,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Union[Generator[str, None, None], str]:
        """
        分析视频内容

        Args:
            video_source: 视频源，支持 本地路径 / http(s) URL / base64: 前缀
            prompt: 分析提示词
            stream: 是否流式输出
            reasoning_effort: "low" / "high" / "max"

        Returns:
            stream=True 返回生成器；否则返回完整响应字符串
        """
        video_b64 = self._load_video(video_source)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{video_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            reasoning_effort=reasoning_effort,
        )

        if stream:
            return self._process_stream(response)
        else:
            return response.choices[0].message.content

    def describe_video(self, video_source: str) -> str:
        """快速描述视频内容"""
        return self.analyze_video(
            video_source,
            prompt="请用2-3句话简洁地描述这个视频的主要内容。",
            stream=False,
        )

    def analyze_timeline(self, video_source: str) -> str:
        """分析视频时间线"""
        return self.analyze_video(
            video_source,
            prompt="请按照时间顺序，详细描述视频中发生的事件，格式为：[时间点] 事件描述",
            stream=False,
        )

    def extract_key_frames(self, video_source: str) -> str:
        """提取关键帧描述"""
        return self.analyze_video(
            video_source,
            prompt="请识别视频中的关键画面/场景，描述每个关键画面的内容和出现的大致时间。",
            stream=False,
        )

    def analyze_emotion(self, video_source: str) -> str:
        """分析视频情感/氛围"""
        return self.analyze_video(
            video_source,
            prompt="请分析这个视频的情感氛围和情绪基调，包括画面色调、音乐、人物表情等方面。",
            stream=False,
        )

    def answer_video_question(self, video_source: str, question: str) -> str:
        """基于视频回答特定问题"""
        return self.analyze_video(
            video_source,
            prompt=f"请观看视频后回答以下问题：{question}",
            stream=False,
        )

    # ============================================================
    # 视频生成功能
    # ============================================================

    def generate_video(
        self,
        prompt: str,
        duration: int = 5,
        size: str = "1280x720",
    ) -> str:
        """
        生成视频

        Args:
            prompt: 视频生成提示词
            duration: 视频时长（秒），默认 5 秒
            size: 视频尺寸，默认 "1280x720"

        Returns:
            生成的视频 URL
        """
        response = self.client.images.generate(
            model="kimi-k3-video",
            prompt=prompt,
            size=size,
            extra_body={"duration": duration},
        )

        if response.data:
            return response.data[0].url
        raise RuntimeError("视频生成失败")

    def generate_and_save_video(
        self,
        prompt: str,
        output_path: str,
        duration: int = 5,
        size: str = "1280x720",
    ) -> str:
        """
        生成视频并保存到本地

        Returns:
            保存的文件路径
        """
        url = self.generate_video(prompt, duration=duration, size=size)

        print("正在下载生成的视频...")

        resp = requests.get(url, stream=True, timeout=300, proxies={})
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(
                        f"\r  下载进度: {pct}% ({downloaded // 1024}KB / {total // 1024}KB)",
                        end="",
                        flush=True,
                    )

        print(f"\n视频已保存: {output_path}")
        return output_path

    def image_to_video(self, image_source: str, prompt: str = "") -> str:
        """
        将图片转换为视频

        Returns:
            生成的视频 URL
        """
        image_b64, _ = self._load_image(image_source)

        full_prompt = f"将这张图片转换为动态视频。{prompt}" if prompt else "将这张图片转换为动态视频"

        response = self.client.images.generate(
            model="kimi-k3-video",
            prompt=full_prompt,
            image=f"data:image/png;base64,{image_b64}",
            extra_body={"type": "image_to_video"},
        )

        if response.data:
            return response.data[0].url
        raise RuntimeError("图片转视频失败")

    # ============================================================
    # 工具方法
    # ============================================================

    def _load_image(self, image_source: str) -> Tuple[str, str]:
        """加载图片并转换为 base64，返回 (base64, mime_type)"""
        if image_source.startswith("base64:"):
            return image_source[7:], "image/png"

        if image_source.startswith(("http://", "https://")):
            return self._download_image(image_source)

        return self._read_local_image(image_source)

    def _download_image(self, url: str) -> Tuple[str, str]:
        """从 URL 下载图片"""
        print(f"正在下载图片: {url}")
        resp = requests.get(url, timeout=120, proxies={})
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "image/png")
        if ";" in content_type:
            content_type = content_type.split(";")[0].strip()

        image_b64 = base64.b64encode(resp.content).decode("utf-8")
        print(f"  下载完成，大小: {len(resp.content) // 1024}KB")
        return image_b64, content_type

    def _read_local_image(self, file_path: str) -> Tuple[str, str]:
        """读取本地图片文件"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {file_path}")

        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(path.suffix.lower(), "image/png")

        print(f"正在读取图片: {file_path}")
        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        print(f"  读取完成，大小: {len(image_b64) * 3 // 4 // 1024}KB")
        return image_b64, mime_type

    def _load_video(self, video_source: str) -> str:
        """加载视频并转换为 base64"""
        if video_source.startswith("base64:"):
            return video_source[7:]

        if video_source.startswith(("http://", "https://")):
            return self._download_video(video_source)

        return self._read_local_file(video_source)

    def _download_video(self, url: str) -> str:
        """从 URL 下载视频"""
        print(f"正在下载视频: {url}")
        resp = requests.get(url, stream=True, timeout=300, proxies={})
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            for chunk in resp.iter_content(chunk_size=8192):
                tmp.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(
                        f"\r  下载进度: {pct}% ({downloaded // 1024}KB / {total // 1024}KB)",
                        end="",
                        flush=True,
                    )

            tmp_path = tmp.name

        print(f"\n  下载完成: {tmp_path}")

        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        os.unlink(tmp_path)
        return video_b64

    def _read_local_file(self, file_path: str) -> str:
        """读取本地视频文件"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"视频文件不存在: {file_path}")

        print(f"正在读取视频: {file_path}")
        with open(path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        print(f"  读取完成，大小: {len(video_b64) * 3 // 4 // 1024}KB")
        return video_b64

    @staticmethod
    def _process_stream(response) -> Generator[str, None, None]:
        """
        处理流式响应（向后兼容）。

        注意：此方法将 reasoning_content 与 content 混合输出。
        如需将「思考过程」与「最终回答」分开，请使用 ChatSession 的流式接口。
        """
        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                yield delta.reasoning_content
            if delta.content:
                yield delta.content


def create_client(api_key: Optional[str] = None, model: str = "kimi-k3") -> KimiClient:
    """便捷函数：创建 Kimi 客户端"""
    return KimiClient(api_key=api_key, model=model)


class ChatSession:
    """
    多轮对话会话。

    K3 被训练为「保留思考历史」模式：多轮对话与工具调用时，必须将从 API 返回的
    完整 assistant 消息（含 reasoning_content 与 tool_calls）原样回传，
    仅回传 content 会丢失上下文。本类在内部自动维护这一完整消息列表。

    支持文本与多模态（图片/视频）混合多轮。
    """

    def __init__(
        self,
        client: KimiClient,
        system: Optional[str] = None,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        model: Optional[str] = None,
    ):
        self.client = client
        self.model = model or client.model
        self.reasoning_effort = reasoning_effort
        self.messages: List[Dict[str, Any]] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    # ---------- 多轮入口 ----------

    def chat(
        self,
        prompt: str,
        stream: bool = False,
    ) -> Union[str, Generator[Tuple[str, str], None, None]]:
        """文本多轮对话。"""
        self.messages.append({"role": "user", "content": prompt})
        return self._run(stream)

    def send_image(
        self,
        image_source: str,
        prompt: str,
        stream: bool = False,
    ) -> Union[str, Generator[Tuple[str, str], None, None]]:
        """发送图片 + 文本，进行多模态多轮对话。"""
        image_b64, mime = self.client._load_image(image_source)
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return self._run(stream)

    def send_video(
        self,
        video_source: str,
        prompt: str,
        stream: bool = False,
    ) -> Union[str, Generator[Tuple[str, str], None, None]]:
        """发送视频 + 文本，进行多模态多轮对话。"""
        video_b64 = self.client._load_video(video_source)
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return self._run(stream)

    # ---------- 内部执行 ----------

    def _run(self, stream: bool):
        if stream:
            return self._stream_response()
        return self._blocking_response()

    def _blocking_response(self) -> str:
        resp = self.client.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            reasoning_effort=self.reasoning_effort,
            stream=False,
        )
        msg = resp.choices[0].message
        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
        }
        # 关键：保留完整 thinking history（reasoning_content / tool_calls）
        if getattr(msg, "reasoning_content", None):
            assistant_msg["reasoning_content"] = msg.reasoning_content
        if getattr(msg, "tool_calls", None):
            assistant_msg["tool_calls"] = msg.tool_calls
        self.messages.append(assistant_msg)
        return msg.content

    def _stream_response(self) -> Generator[Tuple[str, str], None, None]:
        """
        流式多轮。按 (kind, text) 逐块产出：
        - kind="reasoning": 思考过程
        - kind="content": 最终回答
        结束后自动把完整 assistant 消息（含 reasoning_content）存入历史。
        """
        reasoning_parts: List[str] = []
        content_parts: List[str] = []

        stream = self.client.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            reasoning_effort=self.reasoning_effort,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)
                yield ("reasoning", delta.reasoning_content)
            if delta.content:
                content_parts.append(delta.content)
                yield ("content", delta.content)

        assistant_msg = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if reasoning_parts:
            assistant_msg["reasoning_content"] = "".join(reasoning_parts)
        self.messages.append(assistant_msg)

    # ---------- 会话管理 ----------

    def reset(self) -> None:
        """清空对话历史（保留 system 消息，如有）。"""
        self.messages = [m for m in self.messages if m.get("role") == "system"]

    def history(self) -> List[Dict[str, Any]]:
        """返回当前完整消息列表（含 reasoning_content）。"""
        return self.messages


# 向后兼容别名
VideoUnderstanding = KimiClient
