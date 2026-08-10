"""
Kimi K3 视频理解使用示例
展示如何在你的代码中使用视频理解功能。
"""

from video_understanding import VideoUnderstanding, create_client


# 示例1: 基础用法
def example_basic():
    """最简单的使用方式"""
    print("示例1: 基础用法")
    print("-" * 40)

    # 创建客户端（需要设置环境变量 KIMI_API_KEY）
    client = create_client()

    # 快速描述视频
    result = client.describe_video("./橘猫-agent.mp4")
    print(f"视频描述: {result}\n")


# 示例2: 流式输出
def example_stream():
    """流式输出，实时显示结果"""
    print("示例2: 流式输出")
    print("-" * 40)

    client = create_client()

    print("正在分析视频...")
    for chunk in client.analyze_video(
        "./橘猫-agent.mp4",
        prompt="请用生动的语言描述这个视频，像讲故事一样。",
        stream=True
    ):
        print(chunk, end="", flush=True)
    print("\n")


# 示例3: 自定义API密钥
def example_custom_key():
    """使用自定义API密钥"""
    print("示例3: 自定义API密钥")
    print("-" * 40)

    # 直接传入API密钥
    client = VideoUnderstanding(api_key="your_api_key_here")

    result = client.describe_video("./橘猫-agent.mp4")
    print(f"视频描述: {result}\n")


# 示例4: 分析在线视频
def example_online_video():
    """分析在线视频URL"""
    print("示例4: 分析在线视频")
    print("-" * 40)

    client = create_client()

    # 使用GitHub上的视频URL
    video_url = "https://github.com/1982167424-art/Kimi-K3/raw/main/橘猫-agent.mp4"

    result = client.analyze_video(
        video_url,
        prompt="请描述这个视频的内容。",
        stream=False
    )
    print(f"视频描述: {result}\n")


# 示例5: 问答模式
def example_qa():
    """针对视频内容进行问答"""
    print("示例5: 视频问答")
    print("-" * 40)

    client = create_client()

    questions = [
        "视频中主要发生了什么？",
        "视频的主角是谁或什么？",
        "视频的拍摄地点可能在哪里？",
    ]

    for q in questions:
        answer = client.answer_question("./橘猫-agent.mp4", q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


# 示例6: 专业分析
def example_professional():
    """专业角度的视频分析"""
    print("示例6: 专业分析")
    print("-" * 40)

    client = create_client()

    professional_prompt = """请从专业视频制作的角度分析这个视频：

1. 镜头语言：使用了哪些镜头技巧？
2. 剪辑节奏：视频的节奏如何？
3. 声音设计：音效和配乐如何配合画面？
4. 视觉风格：整体的视觉风格是什么？
5. 改进建议：有哪些可以改进的地方？"""

    result = client.analyze_video(
        "./橘猫-agent.mp4",
        prompt=professional_prompt,
        stream=False
    )
    print(f"专业分析:\n{result}\n")


# 示例7: 批量分析
def example_batch():
    """批量分析多个视频"""
    print("示例7: 批量分析")
    print("-" * 40)

    client = create_client()

    video_files = [
        "./橘猫-agent.mp4",
        # 可以添加更多视频文件
    ]

    results = {}
    for video in video_files:
        try:
            desc = client.describe_video(video)
            results[video] = {"status": "success", "description": desc}
        except Exception as e:
            results[video] = {"status": "error", "error": str(e)}

    for video, result in results.items():
        print(f"\n视频: {video}")
        if result["status"] == "success":
            print(f"描述: {result['description']}")
        else:
            print(f"错误: {result['error']}")


# 示例8: 自定义分析器
def example_custom_analyzer():
    """创建自定义分析器"""
    print("示例8: 自定义分析器")
    print("-" * 40)

    class CatVideoAnalyzer(VideoUnderstanding):
        """专门分析猫咪视频的分析器"""

        def analyze_cat_video(self, video_source: str) -> str:
            """分析猫咪视频"""
            prompt = """请以猫咪专家的角度分析这个视频：
1. 这是什么品种的猫？
2. 猫咪的行为特征是什么？
3. 猫咪的情绪状态如何？
4. 视频拍摄的环境如何？"""

            return self.analyze_video(
                video_source,
                prompt=prompt,
                stream=False
            )

    analyzer = CatVideoAnalyzer()
    result = analyzer.analyze_cat_video("./橘猫-agent.mp4")
    print(f"猫咪视频分析:\n{result}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Kimi K3 视频理解使用示例")
    print("=" * 60)
    print()

    # 运行所有示例
    examples = [
        ("1", "基础用法", example_basic),
        ("2", "流式输出", example_stream),
        ("3", "自定义密钥", example_custom_key),
        ("4", "在线视频", example_online_video),
        ("5", "视频问答", example_qa),
        ("6", "专业分析", example_professional),
        ("7", "批量分析", example_batch),
        ("8", "自定义分析器", example_custom_analyzer),
    ]

    print("可用示例:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")

    choice = input("\n请选择要运行的示例 (1-8 或 'all'): ").strip()

    if choice == "all":
        for num, name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n示例 {num} ({name}) 出错: {e}")
    else:
        for num, name, func in examples:
            if choice == num:
                try:
                    func()
                except Exception as e:
                    print(f"\n出错: {e}")
                break
        else:
            print(f"无效的选择: {choice}")
