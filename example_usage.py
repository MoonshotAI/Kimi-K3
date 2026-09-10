"""
Kimi K3 video understanding - usage examples
Shows how to use the video understanding features in your own code.
"""

from video_understanding import VideoUnderstanding, create_client


# Example 1: basic usage
def example_basic():
    """The simplest way to use it"""
    print("Example 1: basic usage")
    print("-" * 40)

    # Create a client (requires the KIMI_API_KEY environment variable)
    client = create_client()

    # Quick video description
    result = client.describe_video("./orange-cat-agent.mp4")
    print(f"Video description: {result}\n")


# Example 2: streaming output
def example_stream():
    """Stream the output, showing results as they arrive"""
    print("Example 2: streaming output")
    print("-" * 40)

    client = create_client()

    print("Analyzing video...")
    for chunk in client.analyze_video(
        "./orange-cat-agent.mp4",
        prompt="Describe this video vividly, like telling a story.",
        stream=True
    ):
        print(chunk, end="", flush=True)
    print("\n")


# Example 3: custom API key
def example_custom_key():
    """Use a custom API key"""
    print("Example 3: custom API key")
    print("-" * 40)

    # Pass the API key directly
    client = VideoUnderstanding(api_key="your_api_key_here")

    result = client.describe_video("./orange-cat-agent.mp4")
    print(f"Video description: {result}\n")


# Example 4: analyse a remote video
def example_online_video():
    """Analyse a video from a URL"""
    print("Example 4: analyse a remote video")
    print("-" * 40)

    client = create_client()

    # Use a video URL hosted on GitHub
    video_url = "https://github.com/CherrieTheShifter/Kimi-K3/raw/main/orange-cat-agent.mp4"

    result = client.analyze_video(
        video_url,
        prompt="Describe the content of this video.",
        stream=False
    )
    print(f"Video description: {result}\n")


# Example 5: question-and-answer mode
def example_qa():
    """Ask questions about the video content"""
    print("Example 5: video Q&A")
    print("-" * 40)

    client = create_client()

    questions = [
        "What mainly happens in the video?",
        "Who or what is the subject of the video?",
        "Where might the video have been filmed?",
    ]

    for q in questions:
        answer = client.answer_question("./orange-cat-agent.mp4", q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


# Example 6: professional analysis
def example_professional():
    """Analyse the video from a professional standpoint"""
    print("Example 6: professional analysis")
    print("-" * 40)

    client = create_client()

    professional_prompt = """Analyse this video from a professional video-production standpoint:

1. Camera work: which shot techniques are used?
2. Editing rhythm: how does the pacing feel?
3. Sound design: how do effects and music support the picture?
4. Visual style: what is the overall look?
5. Suggestions: what could be improved?"""

    result = client.analyze_video(
        "./orange-cat-agent.mp4",
        prompt=professional_prompt,
        stream=False
    )
    print(f"Professional analysis:\n{result}\n")


# Example 7: batch analysis
def example_batch():
    """Analyse several videos in a batch"""
    print("Example 7: batch analysis")
    print("-" * 40)

    client = create_client()

    video_files = [
        "./orange-cat-agent.mp4",
        # Add more video files here
    ]

    results = {}
    for video in video_files:
        try:
            desc = client.describe_video(video)
            results[video] = {"status": "success", "description": desc}
        except Exception as e:
            results[video] = {"status": "error", "error": str(e)}

    for video, result in results.items():
        print(f"\nVideo: {video}")
        if result["status"] == "success":
            print(f"Description: {result['description']}")
        else:
            print(f"Error: {result['error']}")


# Example 8: a custom analyser
def example_custom_analyzer():
    """Create a custom analyser"""
    print("Example 8: custom analyser")
    print("-" * 40)

    class CatVideoAnalyzer(VideoUnderstanding):
        """An analyser specialised for cat videos"""

        def analyze_cat_video(self, video_source: str) -> str:
            """Analyse a cat video"""
            prompt = """Analyse this video as a cat expert would:
1. What breed is the cat?
2. What behavioural traits does it show?
3. What is the cat's emotional state?
4. What is the filming environment like?"""

            return self.analyze_video(
                video_source,
                prompt=prompt,
                stream=False
            )

    analyzer = CatVideoAnalyzer()
    result = analyzer.analyze_cat_video("./orange-cat-agent.mp4")
    print(f"Cat video analysis:\n{result}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Kimi K3 video understanding - usage examples")
    print("=" * 60)
    print()

    # Run all examples
    examples = [
        ("1", "Basic usage", example_basic),
        ("2", "Streaming output", example_stream),
        ("3", "Custom API key", example_custom_key),
        ("4", "Remote video", example_online_video),
        ("5", "Video Q&A", example_qa),
        ("6", "Professional analysis", example_professional),
        ("7", "Batch analysis", example_batch),
        ("8", "Custom analyser", example_custom_analyzer),
    ]

    print("Available examples:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")

    choice = input("\nChoose an example to run (1-8 or 'all'): ").strip()

    if choice == "all":
        for num, name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\nExample {num} ({name}) failed: {e}")
    else:
        for num, name, func in examples:
            if choice == num:
                try:
                    func()
                except Exception as e:
                    print(f"\nError: {e}")
                break
        else:
            print(f"Invalid choice: {choice}")
