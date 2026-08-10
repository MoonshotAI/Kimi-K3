"""
Kimi K3 视频理解 - 快速参考
"""

# ============================================================
# 安装依赖
# ============================================================
# pip3 install openai requests

# ============================================================
# 设置环境变量
# ============================================================
# export KIMI_API_KEY="sk-your-api-key-here"
# API key 从 https://platform.kimi.ai 获取

# ============================================================
# 快速开始
# ============================================================

from video_understanding import VideoUnderstanding, create_client

# 方式1: 使用环境变量
client = create_client()

# 方式2: 直接传入API密钥
client = VideoUnderstanding(api_key="sk-your-api-key")

# ============================================================
# 基础用法
# ============================================================

# 快速描述视频（非流式）
result = client.describe_video("./video.mp4")
print(result)

# 流式输出
for chunk in client.analyze_video(
    "./video.mp4",
    prompt="描述视频内容",
    stream=True
):
    print(chunk, end="", flush=True)

# ============================================================
# 支持的视频源
# ============================================================

# 1. 本地文件
client.describe_video("./my_video.mp4")

# 2. HTTP/HTTPS URL
client.describe_video("https://example.com/video.mp4")

# 3. Base64编码
client.describe_video("base64:IGluZGV4X29mX2Jhc2U2NA==")

# ============================================================
# 预设分析模式
# ============================================================

# 基础描述
client.describe_video("./video.mp4")

# 时间线分析
client.analyze_timeline("./video.mp4")

# 关键帧提取
client.extract_key_frames("./video.mp4")

# 情感分析
client.analyze_emotion("./video.mp4")

# 问答
client.answer_question("./video.mp4", "视频里有什么？")

# ============================================================
# 自定义提示词
# ============================================================

custom_prompt = """
请从以下角度分析视频：
1. 画面内容
2. 声音/音乐
3. 情感氛围
4. 技术亮点
"""

result = client.analyze_video(
    "./video.mp4",
    prompt=custom_prompt,
    stream=False
)
print(result)

# ============================================================
# 高级配置
# ============================================================

# 调整推理强度
client.analyze_video(
    "./video.mp4",
    reasoning_effort="low"    # "low", "medium", "high"
)

# 使用不同的模型
client = VideoUnderstanding(model="kimi-k3")

# ============================================================
# 完整示例
# ============================================================

def analyze_my_video(video_path: str):
    """分析视频的完整示例"""
    client = create_client()

    print("正在分析视频...")

    # 基础描述
    print("\n【快速描述】")
    desc = client.describe_video(video_path)
    print(desc)

    # 详细分析
    print("\n【详细分析】")
    for chunk in client.analyze_video(
        video_path,
        prompt="请详细分析这个视频的各个方面。",
        stream=True
    ):
        print(chunk, end="", flush=True)

    print("\n\n分析完成!")

# 运行示例
# analyze_my_video("./橘猫-agent.mp4")

# ============================================================
# 文件结构
# ============================================================
#
# video_understanding.py  - 核心模块（VideoUnderstanding类）
# test_video_understanding.py - 功能测试脚本
# example_usage.py         - 使用示例集合
# quick_reference.py       - 本文件（快速参考）
#
