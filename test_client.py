"""
Offline tests for retries, token usage, and the CLI parser.

No API key and no network - every external call is stubbed. Runs under
pytest, or standalone with `python test_client.py`.
"""

import sys
import types

import kimi_multimodal as km
from kimi_multimodal import KimiClient, _retry_delay, RETRY_MAX_DELAY


# ============ helpers ============

def _client(**kw):
    """A client with a dummy key, so no real credentials are needed."""
    return KimiClient(api_key="dummy-key-for-test", **kw)


class _FakeResponse:
    def __init__(self, status=200, headers=None, content=b"data"):
        self.status_code = status
        self.headers = headers or {}
        self.content = content
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"raise_for_status called on {self.status_code}")

    def close(self):
        self.closed = True


class _FakeUsage:
    def __init__(self, p, c, r, t):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = t
        self.completion_tokens_details = types.SimpleNamespace(reasoning_tokens=r)


class _FakeCompletion:
    def __init__(self, usage):
        self.usage = usage


# ============ backoff ============

def test_backoff_grows_and_is_capped():
    """Delay increases with attempt number and never exceeds the ceiling."""
    d0 = _retry_delay(0)
    d3 = _retry_delay(3)
    assert 0 < d0 <= 1.0, f"first delay out of range: {d0}"
    assert d3 > d0, "backoff should grow with attempt number"
    assert _retry_delay(50) <= RETRY_MAX_DELAY, "delay must be capped"


def test_retry_after_header_is_honoured():
    """A Retry-After header overrides the computed backoff."""
    assert _retry_delay(0, retry_after="7") == 7.0
    assert _retry_delay(0, retry_after="9999") == RETRY_MAX_DELAY
    # a malformed header falls back to normal backoff rather than crashing
    assert _retry_delay(0, retry_after="soon") > 0


# ============ retries ============

def test_retries_on_429_then_succeeds(monkeypatch=None):
    """A 429 is retried and the eventual 200 is returned."""
    client = _client(max_retries=3)
    calls = []

    def fake_get(url, stream=False, timeout=120, proxies=None):
        calls.append(url)
        if len(calls) < 3:
            return _FakeResponse(429, {"Retry-After": "0"})
        return _FakeResponse(200)

    km.requests.get = fake_get
    km.time.sleep = lambda s: None

    resp = client._get("https://example.com/x")
    assert resp.status_code == 200
    assert len(calls) == 3, f"expected 3 attempts, got {len(calls)}"


def test_gives_up_after_max_retries():
    """Persistent failure raises rather than looping forever."""
    client = _client(max_retries=2)
    calls = []

    def fake_get(url, stream=False, timeout=120, proxies=None):
        calls.append(url)
        raise km.requests.RequestException("connection reset")

    km.requests.get = fake_get
    km.time.sleep = lambda s: None

    try:
        client._get("https://example.com/x")
    except RuntimeError as exc:
        assert "after 3 attempts" in str(exc), str(exc)
    else:
        raise AssertionError("should have raised RuntimeError")
    assert len(calls) == 3, f"expected 3 attempts, got {len(calls)}"


def test_404_is_not_retried():
    """Client errors other than 408/429 fail immediately - a repeat won't help."""
    client = _client(max_retries=3)
    calls = []

    def fake_get(url, stream=False, timeout=120, proxies=None):
        calls.append(url)
        r = _FakeResponse(404)
        r.raise_for_status = lambda: (_ for _ in ()).throw(
            km.requests.HTTPError("404")
        )
        return r

    km.requests.get = fake_get
    km.time.sleep = lambda s: None

    try:
        client._get("https://example.com/missing")
    except km.requests.HTTPError:
        pass
    else:
        raise AssertionError("404 should propagate")
    assert len(calls) == 1, f"404 must not be retried, got {len(calls)} attempts"


# ============ token usage ============

def test_usage_is_recorded_and_accumulates():
    client = _client()
    client._record_usage(_FakeCompletion(_FakeUsage(100, 50, 30, 150)))

    assert client.last_usage["prompt_tokens"] == 100
    assert client.last_usage["reasoning_tokens"] == 30
    assert client.total_usage["calls"] == 1

    client._record_usage(_FakeCompletion(_FakeUsage(10, 5, 3, 15)))
    assert client.last_usage["prompt_tokens"] == 10, "last_usage should be the latest call"
    assert client.total_usage["prompt_tokens"] == 110, "total should accumulate"
    assert client.total_usage["total_tokens"] == 165
    assert client.total_usage["calls"] == 2


def test_usage_survives_a_response_with_no_usage_field():
    client = _client()
    client._record_usage(types.SimpleNamespace())      # no .usage at all
    client._record_usage(_FakeCompletion(None))        # .usage is None
    assert client.total_usage["calls"] == 0, "nothing should be counted"


def test_reset_usage():
    client = _client()
    client._record_usage(_FakeCompletion(_FakeUsage(1, 2, 3, 6)))
    client.reset_usage()
    assert client.last_usage == {}
    assert client.total_usage["calls"] == 0
    assert client.total_usage["total_tokens"] == 0


# ============ CLI parser ============

def test_cli_parses_every_subcommand():
    from kimi_cli import build_parser
    p = build_parser()

    a = p.parse_args(["chat", "hello"])
    assert a.command == "chat" and a.prompt == "hello"
    assert a.effort == "max", "default effort should be max"

    a = p.parse_args(["image", "./x.png", "--ocr"])
    assert a.ocr is True

    a = p.parse_args(["video", "./x.mp4", "--timeline"])
    assert a.timeline is True

    a = p.parse_args(["gen-image", "a cat", "-o", "out.png", "-n", "3"])
    assert a.output == "out.png" and a.n == 3

    a = p.parse_args(["gen-video", "waves", "--duration", "9"])
    assert a.duration == 9

    a = p.parse_args(["--effort", "low", "--max-retries", "5", "chat", "hi"])
    assert a.effort == "low" and a.max_retries == 5


def test_cli_rejects_bad_effort():
    from kimi_cli import build_parser
    try:
        build_parser().parse_args(["--effort", "medium", "chat", "hi"])
    except SystemExit:
        pass
    else:
        raise AssertionError("'medium' is not a valid effort and should be rejected")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
