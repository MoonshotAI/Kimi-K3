"""
ChatSession offline unit tests (no API key or network required).

Verifies:
1. Across turns, the assistant message (including reasoning_content) is kept in history as-is,
   as K3's preserved thinking history requires.
2. The streaming interface yields reasoning and content separately.
3. reset() keeps only the system message.
4. reasoning_effort defaults to "max" and is passed to the underlying API.
"""

from kimi_multimodal import KimiClient, ChatSession


# ============ Mock OpenAI client ============

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
            _FakeMessage(content="This is the answer", reasoning_content="[thinking] let me work through this")
        )

    def _stream(self):
        yield _FakeChunk(_FakeDelta(reasoning_content="[thinking] let me work through this"))
        yield _FakeChunk(_FakeDelta(content="This is the answer"))


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self, *a, **k):
        self.chat = _FakeChat()


def _make_client():
    c = KimiClient(api_key="dummy-key-for-test")
    c.client = _FakeOpenAI()  # swap in the fake client so no real network call happens
    return c


# ============ Test cases ============

def test_blocking_preserves_reasoning_history():
    client = _make_client()
    session = client.create_session(system="You are an assistant")

    # First turn
    out = session.chat("hello")
    assert out == "This is the answer", f"unexpected content: {out}"

    hist = session.history()
    # system + user + assistant
    assert len(hist) == 3, f"history should hold 3 messages, got {len(hist)}"
    assert hist[0] == {"role": "system", "content": "You are an assistant"}
    assert hist[1] == {"role": "user", "content": "hello"}
    # Critical: the assistant message must retain reasoning_content
    assert hist[2]["role"] == "assistant"
    assert hist[2]["content"] == "This is the answer"
    assert hist[2]["reasoning_content"] == "[thinking] let me work through this", "thinking history was not preserved"

    # Second turn: history should keep accumulating
    session.chat("ask again")
    assert len(session.history()) == 5, "history should hold 5 messages after the second turn"


def test_streaming_separates_reasoning_and_content():
    client = _make_client()
    session = client.create_session()

    chunks = list(session.chat("hello", stream=True))
    kinds = [k for k, _ in chunks]
    assert kinds == ["reasoning", "content"], f"unexpected chunk kinds: {kinds}"

    # After streaming, history still holds the complete assistant message (with reasoning)
    assert session.history()[-1]["reasoning_content"] == "[thinking] let me work through this"
    assert session.history()[-1]["content"] == "This is the answer"


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
    # Use a tiny 1x1 png in base64 so no real file is read
    tiny_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLv"
    )
    session.send_image(f"base64:{tiny_png}", "describe this image")
    # send_image appends a user message and calls the API immediately: assistant is last, user is [-2]
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
    print("All tests passed.")
