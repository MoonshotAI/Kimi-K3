# Code audit

This repository is a fork of [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3). Upstream ships four files — a README, a licence, a logo and the technical report. Everything else here arrived through community pull requests from nine different contributors.

Because that is a lot of code from a lot of strangers, every file has been audited. This document records what was checked and what was found, so anyone using this fork can judge it for themselves rather than taking it on trust.

**Last audited:** September 9, 2026

---

## Findings

### API key handling — clean

The key is read from the `KIMI_API_KEY` environment variable, or passed explicitly to `KimiClient(api_key=...)`. It is used in exactly one place: constructing the official `openai.OpenAI` client.

It is never logged, never written to disk, and never sent anywhere other than the configured API base.

### Network calls — clean

Every outbound request was traced. The only hosts contacted are:

| Host | Why |
|---|---|
| `api.moonshot.cn` | The Kimi API itself, via the official `openai` SDK |
| `github.com` | Downloading the sample video in the test scripts |
| `picsum.photos` | Placeholder images for image tests |

All direct calls are `requests.get()` with explicit timeouts. There are no `POST` requests outside the official SDK, and no telemetry or analytics of any kind.

### File operations — clean

The code writes only to paths the caller supplies (`output_path`), plus one temporary file used when downloading a video, which is deleted immediately after reading. Nothing else on disk is touched, and nothing is deleted.

### Code execution — clean

No `eval`, `exec`, `compile`, `__import__`, `pickle`, `marshal`, `subprocess`, `os.system`, or `popen` anywhere in the Python.

`base64` is used for encoding only — images and video are encoded to send to the API. It is never used to decode, so no hidden payload can be smuggled in.

### Tests — passing

Two suites run offline with no API key and no network, against mocked clients.

`test_chat_session.py` verifies that thinking history is preserved across turns, that streaming separates reasoning from content, that `reset()` keeps only the system message, and that `reasoning_effort` defaults to `"max"`.

`test_client.py` verifies the retry policy — that 429 is retried and 404 is not, that backoff grows and is capped, that a `Retry-After` header is honoured, and that the client gives up rather than looping — plus token accounting and every CLI subcommand.

All 15 tests pass. CI runs both on Python 3.9, 3.11 and 3.13 for every push and pull request.

### Retry behaviour — bounded

The retry loop is capped by `max_retries` (default 3) and backs off exponentially with a 30-second ceiling. It cannot spin indefinitely, and it does not retry client errors that would never succeed. Verified by test.

---

## One thing to be aware of

`studio/Unsloth_Studio_Colab.ipynb` clones [unslothai/unsloth](https://github.com/unslothai/unsloth) and runs `studio/setup.sh` from it. That is the official Unsloth repository and the notebook is doing what it says on the tin, but it does mean executing third-party code inside your Colab session. Read the notebook before running it, as you would with any Colab notebook.

---

## Changes made to upstream

Four corrections to the model documentation, all of which are upstream errors:

| | Upstream | Here |
|---|---|---|
| Expert count row | "Number of Experts" | "Number of Routed Experts" |
| Per-token row | "Selected Experts per Token" | "Routed Experts Selected per Token" |
| Modality | "Text, Image" | "Text, Image, Video" |
| Code example | `message.reasoning` | `message.reasoning_content` |

The modality row contradicts upstream's own Key Features, which state the model understands text, images and video, and its own benchmark table, which includes Video-MME and MMVU.

`message.reasoning` is not a field in the OpenAI SDK — upstream's example crashes as written.

Additionally, the contributed code originally contained Chinese comments and docstrings throughout, hardcoded download URLs pointing at a contributor's personal fork, and a `.gitignore` listing files that were already tracked. All three have been fixed.

---

## Scope and limits

This is a static audit: reading the source, tracing data flow, and running the offline test suite. The API-dependent tests were not executed, since that requires a live key.

It reflects the code as of the date above. If you are reading this much later, check the commit history for changes made since.
