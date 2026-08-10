"""
ChatSession 离线单元测试（无需 API key / 网络）。

验证：
1. 多轮对话中，assistant 消息（含 reasoning_content）被原样保留到历史，符合 K3 的 thinking history 要求。
2. 流式接口能把 reasoning 与 content 分开产出。
3. reset() 仅保留 system 消息。
4. 默认 reasoning_effort 为 "max"，且会传给底层 API。
"""

from kimi_multimodal import KimiClient, ChatSession


# ============ Mock OpenAI 客户端 ============

class _FakeDelta:
    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, payload):
        self.message = payload
        self.delta = payload


class _FakeChunk:
    def __init__(self, delta):
        self.choices = [_FakeChoice(delta)]


class _FakeMessage:
    def __init__(self, content, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class _FakeCompletion:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class _FakeCompletions:
    def __init__(self):
        self.calls = 0
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            return self._stream()
        return _FakeCompletion(
            _FakeMessage(content="这是回答", reasoning_content="[思考] 我来分析一下")
        )

    def _stream(self):
        yield _FakeChunk(_FakeDelta(reasoning_content="[思考] 我来分析一下"))
        yield _FakeChunk(_FakeDelta(content="这是回答"))


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self, *a, **k):
        self.chat = _FakeChat()


def _make_client():
    c = KimiClient(api_key="dummy-key-for-test")
    c.client = _FakeOpenAI()  # 替换底层客户端，避免真实网络调用
    return c


# ============ 测试用例 ============

def test_blocking_preserves_reasoning_history():
    client = _make_client()
    session = client.create_session(system="你是一个助手")

    # 第一轮
    out = session.chat("你好")
    assert out == "这是回答", f"返回内容错误: {out}"

    hist = session.history()
    # system + user + assistant
    assert len(hist) == 3, f"历史长度应为3，实际 {len(hist)}"
    assert hist[0] == {"role": "system", "content": "你是一个助手"}
    assert hist[1] == {"role": "user", "content": "你好"}
    # 关键：assistant 消息必须保留 reasoning_content
    assert hist[2]["role"] == "assistant"
    assert hist[2]["content"] == "这是回答"
    assert hist[2]["reasoning_content"] == "[思考] 我来分析一下", "thinking history 未保留！"

    # 第二轮：历史应继续累积
    session.chat("再问一次")
    assert len(session.history()) == 5, "第二轮后历史应为5条"


def test_streaming_separates_reasoning_and_content():
    client = _make_client()
    session = client.create_session()

    chunks = list(session.chat("你好", stream=True))
    kinds = [k for k, _ in chunks]
    assert kinds == ["reasoning", "content"], f"分块类型错误: {kinds}"

    # 流式结束后历史里仍有完整 assistant 消息（含 reasoning）
    assert session.history()[-1]["reasoning_content"] == "[思考] 我来分析一下"
    assert session.history()[-1]["content"] == "这是回答"


def test_reset_keeps_system():
    client = _make_client()
    session = client.create_session(system="SYS")
    session.chat("a")
    session.chat("b")
    session.reset()
    assert session.history() == [{"role": "system", "content": "SYS"}]


def test_default_reasoning_effort_is_max():
    client = _make_client()
    session = client.create_session()
    session.chat("hi")
    assert session.client.client.chat.completions.last_kwargs["reasoning_effort"] == "max"


def test_send_image_appends_multimodal_message():
    client = _make_client()
    session = client.create_session()
    # 用极小的 1x1 png base64，避免真实读取文件
    tiny_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLv"
    )
    session.send_image(f"base64:{tiny_png}", "描述这张图")
    # send_image 会追加 user 消息并立即调用 API，assistant 在末尾，user 在 [-2]
    user_msg = session.history()[-2]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0]["type"] == "image_url"
    assert "base64," in user_msg["content"][0]["image_url"]["url"]


if __name__ == "__main__":
    test_blocking_preserves_reasoning_history()
    test_streaming_separates_reasoning_and_content()
    test_reset_keeps_system()
    test_default_reasoning_effort_is_max()
    test_send_image_appends_multimodal_message()
    print("全部测试通过 ✓")
