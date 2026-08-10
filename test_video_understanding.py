"""
Kimi K3 视频理解测试脚本
测试各种视频理解功能，包括内容描述、时间线分析、情感分析等。
"""

import os
import sys
from video_understanding import VideoUnderstanding

# === 配置 ===
GITHUB_RAW_URL = "https://github.com/1982167424-art/Kimi-K3/raw/main/橘猫-agent.mp4"
LOCAL_VIDEO_PATH = "./橘猫-agent.mp4"


def download_video():
    """下载测试视频"""
    import requests

    if os.path.exists(LOCAL_VIDEO_PATH):
        print(f"[1/3] 视频已存在，跳过下载: {LOCAL_VIDEO_PATH}")
        return

    print(f"[1/3] 正在从 GitHub 下载视频...")
    resp = requests.get(GITHUB_RAW_URL, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(LOCAL_VIDEO_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                print(f"\r  下载进度: {pct}% ({downloaded // 1024}KB / {total // 1024}KB)", end="", flush=True)
    print(f"\n  下载完成: {LOCAL_VIDEO_PATH} ({downloaded // 1024}KB)")


def test_basic_understanding():
    """测试基础视频理解"""
    print("\n" + "=" * 60)
    print("测试 1: 基础视频理解（流式输出）")
    print("=" * 60)

    client = VideoUnderstanding()

    print("\n[流式输出]")
    for chunk in client.analyze_video(
        LOCAL_VIDEO_PATH,
        prompt="请详细描述这个视频的内容，包括画面、动作和氛围。",
        stream=True
    ):
        print(chunk, end="", flush=True)
    print()


def test_quick_describe():
    """测试快速描述"""
    print("\n" + "=" * 60)
    print("测试 2: 快速视频描述")
    print("=" * 60)

    client = VideoUnderstanding()
    result = client.describe_video(LOCAL_VIDEO_PATH)
    print(f"\n{result}")


def test_timeline():
    """测试时间线分析"""
    print("\n" + "=" * 60)
    print("测试 3: 时间线分析")
    print("=" * 60)

    client = VideoUnderstanding()
    result = client.analyze_timeline(LOCAL_VIDEO_PATH)
    print(f"\n{result}")


def test_key_frames():
    """测试关键帧提取"""
    print("\n" + "=" * 60)
    print("测试 4: 关键帧提取")
    print("=" * 60)

    client = VideoUnderstanding()
    result = client.extract_key_frames(LOCAL_VIDEO_PATH)
    print(f"\n{result}")


def test_emotion_analysis():
    """测试情感分析"""
    print("\n" + "=" * 60)
    print("测试 5: 情感/氛围分析")
    print("=" * 60)

    client = VideoUnderstanding()
    result = client.analyze_emotion(LOCAL_VIDEO_PATH)
    print(f"\n{result}")


def test_qa():
    """测试问答"""
    print("\n" + "=" * 60)
    print("测试 6: 视频问答")
    print("=" * 60)

    client = VideoUnderstanding()
    questions = [
        "视频里有几个人？",
        "视频的背景音乐是什么类型的？",
        "视频是在室内还是室外拍摄的？"
    ]

    for q in questions:
        print(f"\n问题: {q}")
        answer = client.answer_question(LOCAL_VIDEO_PATH, q)
        print(f"回答: {answer}")


def test_custom_prompt():
    """测试自定义提示"""
    print("\n" + "=" * 60)
    print("测试 7: 自定义提示词分析")
    print("=" * 60)

    client = VideoUnderstanding()
    custom_prompt = """请以专业的视频分析师角度，从以下几个维度分析这个视频：
1. 画面构图和镜头运用
2. 色彩搭配和视觉风格
3. 叙事结构和节奏
4. 创意亮点和可改进之处"""

    result = client.analyze_video(
        LOCAL_VIDEO_PATH,
        prompt=custom_prompt,
        stream=False
    )
    print(f"\n{result}")


def main():
    """主测试流程"""
    print("=" * 60)
    print("Kimi K3 视频理解功能完整测试")
    print("=" * 60)

    # 下载视频
    download_video()

    # 检查API密钥
    if not os.environ.get("KIMI_API_KEY"):
        print("\n错误: 请先设置环境变量 KIMI_API_KEY")
        print("  export KIMI_API_KEY=\"your_api_key\"")
        print("  API key 从 https://platform.kimi.ai 获取")
        sys.exit(1)

    # 运行测试
    tests = [
        ("1", "基础理解", test_basic_understanding),
        ("2", "快速描述", test_quick_describe),
        ("3", "时间线", test_timeline),
        ("4", "关键帧", test_key_frames),
        ("5", "情感分析", test_emotion_analysis),
        ("6", "问答", test_qa),
        ("7", "自定义提示", test_custom_prompt),
        ("all", "全部测试", None),
    ]

    print("\n可用测试:")
    for num, name, _ in tests:
        print(f"  {num}. {name}")

    choice = input("\n请选择要运行的测试 (输入数字或 'all'): ").strip()

    if choice == "all":
        for num, name, test_func in tests[:-1]:
            try:
                test_func()
            except Exception as e:
                print(f"\n测试 {num} ({name}) 出错: {e}")
    else:
        for num, name, test_func in tests:
            if choice == num and test_func:
                try:
                    test_func()
                except Exception as e:
                    print(f"\n测试出错: {e}")
                break
        else:
            print(f"无效的选择: {choice}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
