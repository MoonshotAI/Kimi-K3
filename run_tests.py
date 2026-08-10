"""
Kimi K3 多模态功能自动测试
"""

import os
import sys
from kimi_multimodal import KimiClient, create_client

# === 配置 ===
GITHUB_RAW_URL = "https://github.com/1982167424-art/Kimi-K3/raw/main/橘猫-agent.mp4"
LOCAL_VIDEO_PATH = "./橘猫-agent.mp4"
TEST_IMAGE_URL = "https://picsum.photos/800/600"


def download_video():
    """下载测试视频"""
    import requests

    if os.path.exists(LOCAL_VIDEO_PATH):
        print(f"[准备] 视频已存在，跳过下载")
        return

    print(f"[准备] 正在从 GitHub 下载视频...")
    resp = requests.get(GITHUB_RAW_URL, stream=True, timeout=120, proxies={})
    resp.raise_for_status()

    with open(LOCAL_VIDEO_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  下载完成: {LOCAL_VIDEO_PATH}")


def run_image_tests():
    """运行所有图片测试"""
    print("\n" + "=" * 60)
    print("图片功能测试")
    print("=" * 60)

    client = create_client()

    # 测试1: 图片识别
    print("\n[测试1] 图片识别 - 基础描述")
    try:
        result = client.describe_image(TEST_IMAGE_URL)
        print(f"✓ 成功: {result[:100]}...")
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试2: 物体检测
    print("\n[测试2] 图片识别 - 物体检测")
    try:
        result = client.identify_objects(TEST_IMAGE_URL)
        print(f"✓ 成功: {result[:100]}...")
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试3: 场景分析
    print("\n[测试3] 图片识别 - 场景分析")
    try:
        result = client.analyze_scene(TEST_IMAGE_URL)
        print(f"✓ 成功: {result[:100]}...")
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试4: 图片问答
    print("\n[测试4] 图片识别 - 问答模式")
    try:
        result = client.answer_image_question(TEST_IMAGE_URL, "图片中主要的颜色是什么？")
        print(f"✓ 成功: {result[:100]}...")
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试5: 图片生成
    print("\n[测试5] 图片生成")
    try:
        urls = client.generate_image("一只可爱的橘猫", size="1024x1024", n=1)
        print(f"✓ 成功: {urls[0][:50]}...")
    except Exception as e:
        print(f"✗ 失败: {e}")


def run_video_tests():
    """运行所有视频测试"""
    print("\n" + "=" * 60)
    print("视频功能测试")
    print("=" * 60)

    client = create_client()

    # 测试6: 视频理解
    print("\n[测试6] 视频理解 - 流式输出")
    try:
        print("正在分析视频...")
        count = 0
        for chunk in client.analyze_video(
            LOCAL_VIDEO_PATH,
            prompt="请用一句话描述这个视频。",
            stream=True
        ):
            print(chunk, end="", flush=True)
            count += 1
            if count > 10:
                break
        print("\n✓ 成功")
    except Exception as e:
        print(f"\n✗ 失败: {e}")

    # 测试7: 视频快速描述
    print("\n[测试7] 视频快速描述")
    try:
        result = client.describe_video(LOCAL_VIDEO_PATH)
        print(f"✓ 成功: {result[:100]}...")
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试8: 视频问答
    print("\n[测试8] 视频问答")
    try:
        result = client.answer_video_question(LOCAL_VIDEO_PATH, "视频里有什么？")
        print(f"✓ 成功: {result[:100]}...")
    except Exception as e:
        print(f"✗ 失败: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Kimi K3 多模态功能自动测试")
    print("=" * 60)

    # 检查API密钥
    if not os.environ.get("KIMI_API_KEY"):
        print("\n错误: 请先设置环境变量 KIMI_API_KEY")
        sys.exit(1)

    # 下载视频
    download_video()

    # 运行测试
    try:
        run_image_tests()
    except Exception as e:
        print(f"\n图片测试出错: {e}")

    try:
        run_video_tests()
    except Exception as e:
        print(f"\n视频测试出错: {e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
