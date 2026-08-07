"""ChatSession / KimiClient 离线单测。

全程 mock openai 客户端，无需 API key 即可运行：
    python3 test_chat_session.py
"""

import os
import sys
import unittest
from unittest import mock

# 确保测试不依赖真实 key
for _k in ("MOONSHOT_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(_k, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kimi_multimodal as km  # noqa: E402
import video_understanding as vu  # noqa: E402


class _Delta:
    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class _Message:
    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls
        # 兼容 openai SDK 的 .reasoning 别名
        self.reasoning = reasoning_content


class _Choice:
    def __init__(self, message):
        self.message = message
        self.delta = message  # 流式/非流式共用


class _Response:
    def __init__(self, message):
        self.choices = [_Choice(message)]


class _StreamChunk:
    def __init__(self, delta):
        self.choices = [_Choice(delta)]


class TestKimiClientConfig(unittest.TestCase):
    """B1: base_url 统一；B2: reasoning_effort 对齐 README，默认 max。"""

    def test_base_url_is_moonshot_cn(self):
        c = km.KimiClient(api_key="fake")
        self.assertEqual(c.base_url, "https://api.moonshot.cn/v1")

    def test_default_reasoning_effort_is_max(self):
        c = km.KimiClient(api_key="fake")
        self.assertEqual(c.reasoning_effort, "max")

    def test_valid_efforts_accepted(self):
        for e in ("low", "high", "max"):
            self.assertEqual(km.KimiClient(api_key="fake", reasoning_effort=e).reasoning_effort, e)

    def test_invalid_effort_rejected(self):
        # 旧代码里不存在的 "medium" 必须被拒
        with self.assertRaises(ValueError):
            km.KimiClient(api_key="fake", reasoning_effort="medium")


class TestChatSessionThinkingHistory(unittest.TestCase):
    """P0: 多轮对话自动保留 reasoning_content / tool_calls。"""

    def _client_with(self, message):
        client = km.KimiClient(api_key="fake")
        client.client = mock.MagicMock()
        client.client.chat.completions.create.return_value = _Response(message)
        return client

    def test_thinking_and_tool_calls_preserved(self):
        msg = _Message(
            content="call done",
            reasoning_content="I should call tool X",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        )
        client = self._client_with(msg)
        session = km.ChatSession(client, system="you are helpful")

        session.ask("please do it")

        assistant = session.messages[-1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"], "call done")
        self.assertEqual(assistant["reasoning_content"], "I should call tool X")
        self.assertEqual(assistant["tool_calls"], msg.tool_calls)

        # 多轮：下一次 ask 应把完整历史（含 thinking）发给 API
        session.ask("again")
        sent = client.client.chat.completions.create.call_args.kwargs["messages"]
        # system + user1 + assistant1(含 reasoning_content/tool_calls) + user2
        self.assertEqual(sent[0]["role"], "system")
        self.assertEqual(sent[1]["role"], "user")
        self.assertEqual(sent[2]["role"], "assistant")
        self.assertEqual(sent[2]["reasoning_content"], "I should call tool X")
        self.assertIn("tool_calls", sent[2])
        self.assertEqual(sent[3]["role"], "user")

    def test_reasoning_content_always_present_key(self):
        # 即便 reasoning_content 为 None，K3 也要求 key 存在以便回传
        client = self._client_with(_Message(content="ok", reasoning_content=None))
        session = km.ChatSession(client)
        session.ask("hi")
        self.assertIn("reasoning_content", session.messages[-1])

    def test_reset_keeps_system_message(self):
        client = self._client_with(_Message(content="ok"))
        session = km.ChatSession(client, system="sys-prompt")
        session.ask("hi")
        self.assertGreater(len(session.messages), 1)
        session.reset()
        self.assertEqual(session.messages, [{"role": "system", "content": "sys-prompt"}])

    def test_reset_without_system_empties(self):
        client = self._client_with(_Message(content="ok"))
        session = km.ChatSession(client)
        session.ask("hi")
        session.reset()
        self.assertEqual(session.messages, [])


class TestStreaming(unittest.TestCase):
    """P0: 流式把 reasoning 与 content 分开产出。"""

    def test_stream_separates_reasoning_and_content(self):
        client = km.KimiClient(api_key="fake")
        client.client = mock.MagicMock()
        client.client.chat.completions.create.return_value = iter([
            _StreamChunk(_Delta(reasoning_content="think1 ")),
            _StreamChunk(_Delta(reasoning_content="think2 ")),
            _StreamChunk(_Delta(content="ans1 ")),
            _StreamChunk(_Delta(content="ans2 ")),
        ])

        events = list(client.stream_chat([{"role": "user", "content": "hi"}]))

        reasoning = "".join(e["delta"] for e in events if e["type"] == "reasoning")
        content = "".join(e["delta"] for e in events if e["type"] == "content")
        self.assertEqual(reasoning, "think1 think2 ")
        self.assertEqual(content, "ans1 ans2 ")
        self.assertEqual([e["type"] for e in events],
                         ["reasoning", "reasoning", "content", "content"])

    def test_stream_ask_preserves_full_assistant(self):
        client = km.KimiClient(api_key="fake")
        client.client = mock.MagicMock()
        client.client.chat.completions.create.return_value = iter([
            _StreamChunk(_Delta(reasoning_content="R")),
            _StreamChunk(_Delta(content="A1")),
            _StreamChunk(_Delta(content="A2")),
        ])
        session = km.ChatSession(client)
        events = list(session.stream_ask("hi"))
        self.assertEqual([e["type"] for e in events], ["reasoning", "content", "content"])
        assistant = session.messages[-1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["reasoning_content"], "R")
        self.assertEqual(assistant["content"], "A1A2")


class TestMultimodalSend(unittest.TestCase):
    """send_image / send_video 组装多模态 content。"""

    def test_send_image_builds_image_url_content(self):
        client = km.KimiClient(api_key="fake")
        client.client = mock.MagicMock()
        client.client.chat.completions.create.return_value = _Response(_Message(content="seen"))
        client.send_image("https://example.com/a.png", "what is this?")
        sent = client.client.chat.completions.create.call_args.kwargs["messages"]
        last = sent[-1]
        self.assertEqual(last["role"], "user")
        types = [p["type"] for p in last["content"]]
        self.assertIn("image_url", types)
        self.assertIn("text", types)

    def test_send_image_path_to_data_url(self):
        client = km.KimiClient(api_key="fake")
        client.client = mock.MagicMock()
        client.client.chat.completions.create.return_value = _Response(_Message(content="ok"))
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # PNG 头
            path = f.name
        try:
            client.send_image(path, "describe")
            sent = client.client.chat.completions.create.call_args.kwargs["messages"]
            url = sent[-1]["content"][0]["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/png;base64,"))
        finally:
            os.unlink(path)


class TestBackwardCompat(unittest.TestCase):
    """B3: video_understanding 重导出，旧 import 路径可用。"""

    def test_video_understanding_reexport_identity(self):
        self.assertIs(vu.VideoUnderstanding, km.VideoUnderstanding)
        self.assertIs(vu.KimiClient, km.KimiClient)
        self.assertIs(vu.ChatSession, km.ChatSession)

    def test_video_understanding_constructs(self):
        v = vu.VideoUnderstanding(api_key="fake")
        self.assertEqual(v.base_url, "https://api.moonshot.cn/v1")
        self.assertEqual(v.reasoning_effort, "max")
        self.assertEqual(v.default_frames, 8)


if __name__ == "__main__":
    unittest.main()
