# Kimi K3 Python SDK —— 可新增功能清单

> 分析对象：本地 `/Users/zhang/Desktop/tom` 下的 Python 实现（Kimi K3 多模态 API 客户端）。
> 现状：`kimi_multimodal.py` 已实现图片/视频的理解与生成；`video_understanding.py` 是旧版视频模块（与前者功能重复）。
> 优先级：P0 核心补全/修复 → P1 对齐 README 能力 → P2 工程化体验 → P3 仓库与发布。

---

## 先看几个必须修的硬伤（不是新功能，但不修会埋坑）

| # | 问题 | 位置 | 说明 |
|---|------|------|------|
| B1 | **API base 不一致** | `kimi_multimodal.py` 用 `api.moonshot.cn/v1`，`video_understanding.py` 用 `api.moonshot.ai/v1` | 两个 host 只有一个是对的，另一个必失败。需统一。 |
| B2 | **reasoning_effort 取值错误** | 两个模块都写 `"low"/"medium"/"high"` | README 明确写的是 `"low"/"high"/"max"`（默认 `max`）。代码缺 `max`（官方推荐默认值），且多了不存在的 `medium`。 |
| B3 | **两套客户端功能重复** | `kimi_multimodal.py` vs `video_understanding.py` | 视频功能几乎一模一样地写了两遍，维护成本翻倍，且行为不一致。应合并为单一 `KimiClient`。 |

---

## P0 —— 核心能力补全（K3 的关键使用模式，目前完全缺失）

### 1. 多轮对话 + thinking history 保留
K3 被训练成"保留思考历史"模式，**多轮对话和工具调用时必须把 API 返回的完整 assistant 消息（含 `reasoning_content` 和 `tool_calls`）原样回传**，只回传 `content` 会丢上下文。
- 现状：`analyze_image/analyze_video` 都是一次性调用，没有任何会话状态管理。
- 建议：新增 `ChatSession` 类或 `client.chat()`，内部维护 `messages` 列表，自动保留 `reasoning_content`；提供 `reset()`。

### 2. 基础文本对话 `chat()`
- 现状：客户端叫"多模态"，但其实只有图/视频分析和生成，**没有纯文本聊天入口**。
- 建议：`chat(prompt, stream=...)` 走 `chat.completions`，复用同一套 reasoning/stream 逻辑。

### 3. 流式 reasoning 与 content 分离
- 现状：`_process_stream` 把 `reasoning_content` 和 `content` 混在一起 yield。
- 建议：支持回调/分别捕获，便于 UI 把"思考过程"和"最终答案"分开展示（README 强调 K3 始终返回 reasoning）。

### 4. 合并两套客户端，统一 base
- 删除或降级 `video_understanding.py`，所有能力收归 `KimiClient`，base 统一（以官方 README 的 `platform.kimi.ai` 为准并核对实际可用 host）。

---

## P1 —— 对齐 README 已声明但未实现的能力

### 5. 工具调用 / Function Calling（tool_calls）
README 明确 K3 支持工具调用。建议实现 `client.chat_with_tools(tools, ...)` 并处理多轮 tool 循环。

### 6. 结构化输出 / JSON mode
README 提到 structured output。建议 `client.structured(prompt, schema)` 返回解析后的对象。

### 7. 上下文缓存（Context Caching）
README 提到 context caching（长上下文省 token）。建议对长 system prompt / 大视频描述做 cache 标记。

### 8. 异步客户端
所有调用目前是同步 OpenAI client。建议加 `AsyncKimiClient`（`AsyncOpenAI`），便于高并发批量处理。

---

## P2 —— 工程化与体验增强

### 9. 真正的 CLI（替代现在的 input() 菜单）
用 `argparse` 做 `kimi-k3` 命令：
```
kimi-k3 chat "你好"
kimi-k3 image ./cat.png --prompt "描述"
kimi-k3 video ./clip.mp4 --timeline
kimi-k3 gen-image "一只橘猫" -o out.png
```
现在的 `example_usage.py` / `run_tests.py` 都是 `input()` 交互，没法脚本化。

### 10. 批量 / 目录处理 + 并发
把 `example_usage.py` 里手写的 `for` 循环升级为 `batch_analyze(pattern, max_concurrency=4)`，结合 P8 的异步客户端。

### 11. 大文件上传方案（避免全量 base64）
- 现状：视频整个读成 base64 内联进 `data:video/mp4;base64,...`，大视频极易超 token/请求体上限。
- 建议：支持文件上传接口或先传后引用 URL，并兼容更多视频格式（不止 mp4）。

### 12. 重试 / 限流 / 超时与退避
- 现状：无重试，`requests.get(..., timeout=...)` 单次，遇到 429 直接挂。
- 建议：加指数退避重试（429/5xx），可配置超时与最大重试次数。

### 13. Token 用量与成本统计
- 现状：响应里的 `usage` 完全没暴露。
- 建议：在返回结果里附上 `prompt_tokens / completion_tokens`，可选累计统计。

---

## P3 —— 仓库与发布（让 GitHub 仓库不只是文档）

### 14. 把 SDK 纳入 GitHub 仓库
目前仓库只有文档没代码。建议新增 `sdk/` 目录放 Python 实现，并补一份 SDK 自己的 README（区别于模型介绍 README）。

### 15. 打包发布
加 `pyproject.toml` + `requirements.txt`，支持 `pip install kimi-k3-sdk`，并固定 `openai` / `requests` / `httpx` 版本。

### 16. 非交互测试 + CI
- 现状：测试脚本全靠 `input()` 选菜单，无法在 CI 跑。
- 建议：改 pytest，支持 `pytest --api-key=...` 或环境变量，加 `--all` 非交互模式。

### 17. 多模型/能力自动发现
加 `client.list_models()` 或配置中心，方便以后 K3 出子模型时不用改代码。

---

## 建议的实施顺序

1. 先修 B1/B2/B3（统一客户端 + 修正参数），这是地基。
2. 做 P0-1/2/3（多轮对话 + chat + 流式分离），补齐 K3 最关键的使用模式。
3. 再做 P1（工具调用 / 结构化 / 缓存 / 异步）。
4. 最后 P2/P3 提升体验和发布质量。

需要的话，我可以直接从 **B1/B2/B3 + P0** 动手，把 `kimi_multimodal.py` 重构成带 `chat()` 和会话管理的单一客户端。
