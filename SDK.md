# Kimi K3 Python SDK

A small Python client for the Kimi K3 API — text chat, image and video understanding, image and video generation, and multi-turn conversations with preserved thinking history.

> This is the SDK documentation. For the model itself, see [README.md](README.md)

---

## Install

```bash
pip install -r requirements.txt
```

Or install the package itself, which also puts a `kimi-k3` command on your PATH:

```bash
pip install -e .
```

## Set your API key

```bash
export KIMI_API_KEY="your_api_key"
```

Get a key at [platform.kimi.ai](https://platform.kimi.ai).

## Quick start

```python
from kimi_multimodal import KimiClient

client = KimiClient()

print(client.chat("Explain mixture-of-experts in one paragraph."))
print(client.describe_image("./photo.png"))
print(client.describe_video("./orange-cat-agent.mp4"))
```

---

## Multi-turn chat

K3 is trained in preserved-thinking-history mode: the complete assistant message — including `reasoning_content` — has to be passed back on every turn, or context is lost. `ChatSession` handles that for you.

```python
session = client.create_session(system="You are a helpful assistant.")

session.chat("Tell me three random numbers.")
session.chat("What were the other two you had in mind?")   # answers from the preserved reasoning
```

| Method | Purpose |
|---|---|
| `chat(prompt, stream=False)` | Text turn |
| `send_image(image_source, prompt)` | Image + text turn |
| `send_video(video_source, prompt)` | Video + text turn |
| `reset()` | Clear history, keep the system message |
| `history()` | Full message list, reasoning included |

## Streaming

Streaming yields `(kind, text)` pairs, so thinking and the final answer can be shown separately:

```python
for kind, text in session.chat("Why is the sky blue?", stream=True):
    if kind == "reasoning":
        print(text, end="")      # the model's thinking
    else:
        print(text, end="")      # the final answer
```

---

## Command line

Installing the package gives you a `kimi-k3` command. Without installing, run `python kimi_cli.py` instead.

```bash
kimi-k3 chat "Explain mixture-of-experts."
kimi-k3 image ./photo.png --ocr
kimi-k3 video ./clip.mp4 --timeline
kimi-k3 gen-image "a ginger cat" -o out.png
kimi-k3 gen-video "waves at sunset" -o out.mp4
```

Global flags: `--model`, `--effort {low,high,max}`, `--max-retries`, `--usage`. Every subcommand takes `--help`.

## Retries

Connection errors and HTTP 408, 429 and 5xx are retried with exponential backoff and jitter. A `Retry-After` header is honoured when the server sends one. Other 4xx errors fail immediately, since repeating them would not help.

```python
client = KimiClient(max_retries=5)     # default is 3
```

## Token usage

```python
client.chat("hello")

client.last_usage    # {'prompt_tokens': 12, 'completion_tokens': 40,
                     #  'reasoning_tokens': 31, 'total_tokens': 52}
client.total_usage   # the same keys, accumulated, plus 'calls'
client.reset_usage()
```

From the CLI, `--usage` prints the counts to stderr, so they stay out of piped output.

---

## What KimiClient can do

**Images**

`analyze_image` · `describe_image` · `identify_objects` · `analyze_scene` · `extract_text_from_image` (OCR) · `answer_image_question` · `generate_image` · `generate_and_save_image` · `image_to_video`

**Video**

`analyze_video` · `describe_video` · `analyze_timeline` · `extract_key_frames` · `analyze_emotion` · `answer_video_question` · `generate_video` · `generate_and_save_video`

Image and video sources accept a local path, an `http(s)` URL, or a `base64:` prefix.

**Reasoning effort** — `"low"`, `"high"`, or `"max"` (default `"max"`), passed to any call.

---

## Files

| File | What it is |
|---|---|
| `kimi_multimodal.py` | The client. `KimiClient` and `ChatSession` live here. |
| `video_understanding.py` | Backwards-compatible re-export of the old `VideoUnderstanding` class. |
| `kimi_cli.py` | Command-line interface. |
| `kimik3.py` | Minimal standalone example of preserved thinking history. |
| `example_usage.py` | Eight worked examples — basic, streaming, batch, custom analysers. |
| `quick_reference.py` | Cheat sheet of common calls. |
| `run_tests.py` | Quick non-interactive smoke test of image and video features. |
| `test_all_features.py` | Full interactive test menu, 13 tests. |
| `test_video_understanding.py` | Video-focused test menu, 7 tests. |
| `test_chat_session.py` | Offline unit tests for ChatSession — no API key or network needed. |
| `test_client.py` | Offline unit tests for retries, token usage and the CLI parser. |
| `pyproject.toml` · `requirements.txt` | Packaging and dependencies. |
| `.github/workflows/tests.yml` | CI: runs the offline suites on Python 3.9, 3.11 and 3.13. |
| `studio/` | Unsloth Studio Colab notebook. |
| `FEATURE_SUGGESTIONS.md` | Known gaps and a proposed roadmap. |

## Running the tests

```bash
pytest                          # both offline suites, no API key needed
python test_chat_session.py     # or run either standalone
python test_client.py

python run_tests.py             # smoke test, needs KIMI_API_KEY
python test_all_features.py     # interactive menu
```

---

## Known issues

See [FEATURE_SUGGESTIONS.md](FEATURE_SUGGESTIONS.md). The main ones:

- **API base.** The client uses `api.moonshot.cn/v1`; the model README points at `platform.kimi.ai`. Verify which host works for your account.
- **No tool calling, structured output or context caching** yet, though K3 supports all three.
- **No async client**, so batch work runs sequentially.
- **Video is inlined as base64**, so large files can exceed request limits.
