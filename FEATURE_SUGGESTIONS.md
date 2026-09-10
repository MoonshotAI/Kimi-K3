# Kimi K3 Python SDK — roadmap

What has been built, and what is still open.

**Last reviewed:** September 9, 2026

---

## Shipped

### Correctness

- **Unified API base.** The client had two implementations pointing at different hosts (`api.moonshot.cn` and `api.moonshot.ai`). Now one base URL, one client.
- **Correct `reasoning_effort` values.** Was `"low"/"medium"/"high"`; the API accepts `"low"/"high"/"max"` with `"max"` as the default.
- **One client, not two.** `video_understanding.py` duplicated the whole video implementation. It is now a thin re-export so existing imports keep working, and all functionality lives in `KimiClient`.

### Core capability

- **Multi-turn chat with preserved thinking history.** `ChatSession` keeps the complete assistant message — `reasoning_content` and `tool_calls` included — so context survives across turns. Covered by offline tests.
- **Plain text chat.** `client.chat(prompt)` and `session.chat(prompt)`.
- **Streaming that separates thinking from answer.** Streaming yields `(kind, text)` pairs where `kind` is `"reasoning"` or `"content"`, so a UI can show them apart.

### Engineering

- **Retries with backoff.** Connection errors and 408/429/5xx are retried with exponential backoff and jitter; a `Retry-After` header is honoured when the server sends one. Other 4xx errors fail immediately, since repeating them will not help. Configurable via `max_retries`.
- **Token usage.** `client.last_usage` holds the most recent call's counts; `client.total_usage` accumulates across calls, including reasoning tokens. `reset_usage()` clears them. The CLI prints them with `--usage`.
- **A real CLI.** `kimi_cli.py` gives `chat`, `image`, `video`, `gen-image` and `gen-video` subcommands with argparse, replacing the `input()` menus for scripted use.
- **Packaging.** `pyproject.toml` and `requirements.txt`, with a `kimi-k3` console entry point and pinned dependency ranges.
- **Tests and CI.** `test_client.py` and `test_chat_session.py` run offline against mocked clients — no API key, no network. GitHub Actions runs them on Python 3.9, 3.11 and 3.13 on every push and pull request.

---

## Open

### Capabilities the model supports but the client does not

**Tool calling.** K3 supports `tool_calls`. Needs `client.chat_with_tools(tools, ...)` and a multi-turn tool loop. `ChatSession` already preserves `tool_calls` in history, so the groundwork is there.

**Structured output.** K3 supports JSON schema responses. Needs `client.structured(prompt, schema)` returning a parsed object.

**Context caching.** K3 caches long prompts at a large discount. Long system prompts and video descriptions should be marked for caching.

**Async client.** Everything is synchronous. An `AsyncKimiClient` built on `AsyncOpenAI` would make batch work practical.

### Engineering

**Batch processing with concurrency.** `example_usage.py` loops sequentially. A `batch_analyze(pattern, max_concurrency=4)` on top of an async client would be the natural shape.

**Large-file uploads.** Video is read entirely into base64 and inlined as a data URI, so large files can exceed request-body limits. A file-upload endpoint, or upload-then-reference-by-URL, would fix it — and would allow formats beyond mp4.

**Model discovery.** `client.list_models()` or a config source, so new K3 variants do not require code changes.

---

## Notes

The API-backed test scripts — `run_tests.py`, `test_all_features.py`, `test_video_understanding.py` — are interactive and need a live key, so CI skips them. They are still the right way to check real behaviour before a release.

One open question: the client uses `api.moonshot.cn/v1` while the model README points at `platform.kimi.ai`. Both appear in official documentation. Verify which is correct for your account and region.
