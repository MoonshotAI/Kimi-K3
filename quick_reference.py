"""
Kimi K3 video understanding - quick reference
"""

# ============================================================
# Install dependencies
# ============================================================
# pip3 install openai requests

# ============================================================
# Set the environment variable
# ============================================================
# export KIMI_API_KEY="sk-your-api-key-here"
# Get an API key at https://platform.kimi.ai

# ============================================================
# Quick start
# ============================================================

from video_understanding import VideoUnderstanding, create_client

# Option 1: use the environment variable
client = create_client()

# Option 2: pass the API key directly
client = VideoUnderstanding(api_key="sk-your-api-key")

# ============================================================
# Basic usage
# ============================================================

# Quick video description (non-streaming)
result = client.describe_video("./video.mp4")
print(result)

# Streaming output
for chunk in client.analyze_video(
    "./video.mp4",
    prompt="Describe the video content",
    stream=True
):
    print(chunk, end="", flush=True)

# ============================================================
# Supported video sources
# ============================================================

# 1. Local file
client.describe_video("./my_video.mp4")

# 2. HTTP/HTTPS URL
client.describe_video("https://example.com/video.mp4")

# 3. Base64-encoded
client.describe_video("base64:IGluZGV4X29mX2Jhc2U2NA==")

# ============================================================
# Preset analysis modes
# ============================================================

# Basic description
client.describe_video("./video.mp4")

# Timeline analysis
client.analyze_timeline("./video.mp4")

# Key-frame extraction
client.extract_key_frames("./video.mp4")

# Sentiment analysis
client.analyze_emotion("./video.mp4")

# Q&A
client.answer_question("./video.mp4", "What is in the video?")

# ============================================================
# Custom prompt
# ============================================================

custom_prompt = """
Analyse the video from these angles:
1. On-screen content
2. Sound and music
3. Mood and atmosphere
4. Technical highlights
"""

result = client.analyze_video(
    "./video.mp4",
    prompt=custom_prompt,
    stream=False
)
print(result)

# ============================================================
# Advanced configuration
# ============================================================

# Adjust the reasoning effort
client.analyze_video(
    "./video.mp4",
    reasoning_effort="low"    # "low", "medium", "high"
)

# Use a different model
client = VideoUnderstanding(model="kimi-k3")

# ============================================================
# Full example
# ============================================================

def analyze_my_video(video_path: str):
    """A complete video-analysis example"""
    client = create_client()

    print("Analyzing video...")

    # Basic description
    print("\n[Quick description]")
    desc = client.describe_video(video_path)
    print(desc)

    # Detailed analysis
    print("\n[Detailed analysis]")
    for chunk in client.analyze_video(
        video_path,
        prompt="Analyse every aspect of this video in detail.",
        stream=True
    ):
        print(chunk, end="", flush=True)

    print("\n\nAnalysis complete.")

# Run the example
# analyze_my_video("./orange-cat-agent.mp4")

# ============================================================
# File layout
# ============================================================
#
# video_understanding.py  - core module (VideoUnderstanding class)
# test_video_understanding.py - feature test script
# example_usage.py         - collection of usage examples
# quick_reference.py       - this file (quick reference)
#
