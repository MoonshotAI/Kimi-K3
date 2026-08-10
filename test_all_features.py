"""
Kimi K3 多模态功能完整测试
测试图片识别、图片生成、视频理解、视频生成功能。
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
    resp = requests.get(GITHUB_RAW_URL, stream=True, timeout=120)
    resp.raise_for_status()

    with open(LOCAL_VIDEO_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  下载完成: {LOCAL_VIDEO_PATH}")


# ============================================================
# 图片识别测试
# ============================================================

def test_image_recognition():
    """测试图片识别功能"""
    print("\n" + "=" * 60)
    print("测试 1: 图片识别 - 基础描述")
    print("=" * 60)

    client = create_client()
    result = client.describe_image(TEST_IMAGE_URL)
    print(f"\n{result}")


def test_image_objects():
    """测试物体识别"""
    print("\n" + "=" * 60)
    print("测试 2: 图片识别 - 物体检测")
    print("=" * 60)

    client = create_client()
    result = client.identify_objects(TEST_IMAGE_URL)
    print(f"\n{result}")


def test_image_scene():
    """测试场景分析"""
    print("\n" + "=" * 60)
    print("测试 3: 图片识别 - 场景分析")
    print("=" * 60)

    client = create_client()
    result = client.analyze_scene(TEST_IMAGE_URL)
    print(f"\n{result}")


def test_image_qa():
    """测试图片问答"""
    print("\n" + "=" * 60)
    print("测试 4: 图片识别 - 问答模式")
    print("=" * 60)

    client = create_client()
    questions = [
        "图片中主要的颜色是什么？",
        "图片中有哪些物体？",
        "这张图片给人什么感觉？"
    ]

    for q in questions:
        print(f"\n问题: {q}")
        answer = client.answer_image_question(TEST_IMAGE_URL, q)
        print(f"回答: {answer}")


# ============================================================
# 图片生成测试
# ============================================================

def test_image_generation():
    """测试图片生成功能"""
    print("\n" + "=" * 60)
    print("测试 5: 图片生成")
    print("=" * 60)

    client = create_client()

    prompt = "一只可爱的橘猫在阳光下打盹，温馨的场景，高清摄影风格"
    print(f"生成提示: {prompt}")

    urls = client.generate_image(prompt, size="1024x1024", n=1)
    print(f"\n生成成功！")
    for i, url in enumerate(urls, 1):
        print(f"  图片 {i}: {url}")


def test_image_generation_and_save():
    """测试图片生成并保存"""
    print("\n" + "=" * 60)
    print("测试 6: 图片生成并保存到本地")
    print("=" * 60)

    client = create_client()

    prompt = "一个未来城市的夜景，霓虹灯闪烁，科幻风格"
    output_path = "./generated_image.png"

    print(f"生成提示: {prompt}")
    client.generate_and_save_image(prompt, output_path, size="1024x1024")
    print(f"图片已保存: {output_path}")


def test_image_styles():
    """测试不同风格的图片生成"""
    print("\n" + "=" * 60)
    print("测试 7: 不同风格的图片生成")
    print("=" * 60)

    client = create_client()

    styles = [
        ("写实摄影", "一只橘猫，真实照片风格"),
        ("卡通动漫", "一只橘猫，卡通动漫风格"),
        ("油画风格", "一只橘猫，梵高油画风格"),
    ]

    for style_name, prompt in styles:
        print(f"\n风格: {style_name}")
        urls = client.generate_image(prompt, size="1024x1024")
        print(f"  生成成功: {urls[0][:50]}...")


# ============================================================
# 视频理解测试
# ============================================================

def test_video_understanding():
    """测试视频理解功能"""
    print("\n" + "=" * 60)
    print("测试 8: 视频理解 - 流式输出")
    print("=" * 60)

    client = create_client()

    print("\n正在分析视频...")
    for chunk in client.analyze_video(
        LOCAL_VIDEO_PATH,
        prompt="请详细描述这个视频的内容。",
        stream=True
    ):
        print(chunk, end="", flush=True)
    print()


def test_video_describe():
    """测试视频快速描述"""
    print("\n" + "=" * 60)
    print("测试 9: 视频快速描述")
    print("=" * 60)

    client = create_client()
    result = client.describe_video(LOCAL_VIDEO_PATH)
    print(f"\n{result}")


def test_video_timeline():
    """测试视频时间线分析"""
    print("\n" + "=" * 60)
    print("测试 10: 视频时间线分析")
    print("=" * 60)

    client = create_client()
    result = client.analyze_timeline(LOCAL_VIDEO_PATH)
    print(f"\n{result}")


def test_video_qa():
    """测试视频问答"""
    print("\n" + "=" * 60)
    print("测试 11: 视频问答")
    print("=" * 60)

    client = create_client()
    questions = [
        "视频里有几个人？",
        "视频的背景音乐是什么类型的？",
        "视频是在室内还是室外拍摄的？"
    ]

    for q in questions:
        print(f"\n问题: {q}")
        answer = client.answer_video_question(LOCAL_VIDEO_PATH, q)
        print(f"回答: {answer}")


# ============================================================
# 视频生成测试
# ============================================================

def test_video_generation():
    """测试视频生成功能"""
    print("\n" + "=" * 60)
    print("测试 12: 视频生成")
    print("=" * 60)

    client = create_client()

    prompt = "一只橘猫在草地上玩耍，阳光明媚，温馨可爱"
    print(f"生成提示: {prompt}")

    url = client.generate_video(prompt, duration=5, size="1280x720")
    print(f"\n生成成功！")
    print(f"  视频URL: {url}")


def test_video_generation_and_save():
    """测试视频生成并保存"""
    print("\n" + "=" * 60)
    print("测试 13: 视频生成并保存到本地")
    print("=" * 60)

    client = create_client()

    prompt = "夕阳下的海滩，海浪轻轻拍打沙滩"
    output_path = "./generated_video.mp4"

    print(f"生成提示: {prompt}")
    client.generate_and_save_video(prompt, output_path, duration=5)
    print(f"视频已保存: {output_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    """主测试流程"""
    print("=" * 60)
    print("Kimi K3 多模态功能完整测试")
    print("=" * 60)

    # 检查API密钥
    if not os.environ.get("KIMI_API_KEY"):
        print("\n错误: 请先设置环境变量 KIMI_API_KEY")
        print("  export KIMI_API_KEY=\"your_api_key\"")
        print("  API key 从 https://platform.kimi.ai 获取")
        sys.exit(1)

    # 下载视频
    download_video()

    # 测试菜单
    tests = [
        ("1", "图片识别 - 基础描述", test_image_recognition),
        ("2", "图片识别 - 物体检测", test_image_objects),
        ("3", "图片识别 - 场景分析", test_image_scene),
        ("4", "图片识别 - 问答模式", test_image_qa),
        ("5", "图片生成", test_image_generation),
        ("6", "图片生成并保存", test_image_generation_and_save),
        ("7", "图片生成 - 不同风格", test_image_styles),
        ("8", "视频理解 - 流式输出", test_video_understanding),
        ("9", "视频快速描述", test_video_describe),
        ("10", "视频时间线分析", test_video_timeline),
        ("11", "视频问答", test_video_qa),
        ("12", "视频生成", test_video_generation),
        ("13", "视频生成并保存", test_video_generation_and_save),
        ("img", "全部图片测试", None),
        ("vid", "全部视频测试", None),
        ("all", "全部测试", None),
    ]

    print("\n可用测试:")
    for num, name, _ in tests:
        print(f"  {num}. {name}")

    choice = input("\n请选择要运行的测试 (输入数字或 'all'): ").strip()

    if choice == "all":
        for num, name, test_func in tests[:-3]:
            if test_func:
                try:
                    test_func()
                except Exception as e:
                    print(f"\n测试 {num} ({name}) 出错: {e}")
    elif choice == "img":
        for test_func in [test_image_recognition, test_image_objects, test_image_scene, test_image_qa,
                         test_image_generation, test_image_generation_and_save, test_image_styles]:
            try:
                test_func()
            except Exception as e:
                print(f"\n图片测试出错: {e}")
    elif choice == "vid":
        for test_func in [test_video_understanding, test_video_describe, test_video_timeline, test_video_qa,
                         test_video_generation, test_video_generation_and_save]:
            try:
                test_func()
            except Exception as e:
                print(f"\n视频测试出错: {e}")
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
