"""
Kimi K3 video understanding - test script
Tests the video understanding features: description, timeline analysis, sentiment analysis, and more.
"""

import os
import sys
from video_understanding import VideoUnderstanding

# === Configuration ===
GITHUB_RAW_URL = "https://github.com/CherrieTheShifter/Kimi-K3/raw/main/orange-cat-agent.mp4"
LOCAL_VIDEO_PATH = "./orange-cat-agent.mp4"


def download_video():
    """Download the test video"""
    import requests

    if os.path.exists(LOCAL_VIDEO_PATH):
        print(f"[1/3] Video already present, skipping download: {LOCAL_VIDEO_PATH}")
        return

    print(f"[1/3] Downloading video from GitHub...")
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
                print(f"\r  Download progress: {pct}% ({downloaded // 1024}KB / {total // 1024}KB)", end="", flush=True)
    print(f"\n  Download complete: {LOCAL_VIDEO_PATH} ({downloaded // 1024}KB)")


def test_basic_understanding():
    """Test basic video understanding"""
    print("\n" + "=" * 60)
    print("Test 1: basic video understanding (streaming)")
    print("=" * 60)

    client = VideoUnderstanding()

    print("\n[streaming output]")
    for chunk in client.analyze_video(
        LOCAL_VIDEO_PATH,
        prompt="Describe this video in detail: the imagery, the action, and the atmosphere.",
        stream=True
    ):
        print(chunk, end="", flush=True)
    print()


def test_quick_describe():
    """Test quick description"""
    print("\n" + "=" * 60)
    print("Test 2: quick video description")
    print("=" * 60)

    client = VideoUnderstanding()
    result = client.describe_video(LOCAL_VIDEO_PATH)
    print(f"\n{result}")


def test_timeline():
    """Test timeline analysis"""
    print("\n" + "=" * 60)
    print("Test 3: timeline analysis")
    print("=" * 60)

    client = VideoUnderstanding()
    result = client.analyze_timeline(LOCAL_VIDEO_PATH)
    print(f"\n{result}")


def test_key_frames():
    """Test key-frame extraction"""
    print("\n" + "=" * 60)
    print("Test 4: key-frame extraction")
    print("=" * 60)

    client = VideoUnderstanding()
    result = client.extract_key_frames(LOCAL_VIDEO_PATH)
    print(f"\n{result}")


def test_emotion_analysis():
    """Test sentiment analysis"""
    print("\n" + "=" * 60)
    print("Test 5: sentiment / atmosphere analysis")
    print("=" * 60)

    client = VideoUnderstanding()
    result = client.analyze_emotion(LOCAL_VIDEO_PATH)
    print(f"\n{result}")


def test_qa():
    """Test Q&A"""
    print("\n" + "=" * 60)
    print("Test 6: video Q&A")
    print("=" * 60)

    client = VideoUnderstanding()
    questions = [
        "How many people are in the video?",
        "What kind of background music does it use?",
        "Was it filmed indoors or outdoors?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer = client.answer_question(LOCAL_VIDEO_PATH, q)
        print(f"Answer: {answer}")


def test_custom_prompt():
    """Test a custom prompt"""
    print("\n" + "=" * 60)
    print("Test 7: analysis with a custom prompt")
    print("=" * 60)

    client = VideoUnderstanding()
    custom_prompt = """As a professional video analyst, assess this video across these dimensions:
1. Composition and camera work
2. Colour palette and visual style
3. Narrative structure and pacing
4. Creative highlights and areas to improve"""

    result = client.analyze_video(
        LOCAL_VIDEO_PATH,
        prompt=custom_prompt,
        stream=False
    )
    print(f"\n{result}")


def main():
    """Main test flow"""
    print("=" * 60)
    print("Kimi K3 video understanding - full tests")
    print("=" * 60)

    # Download the video
    download_video()

    # Check the API key
    if not os.environ.get("KIMI_API_KEY"):
        print("\nError: set the KIMI_API_KEY environment variable first")
        print("  export KIMI_API_KEY=\"your_api_key\"")
        print("  Get an API key at https://platform.kimi.ai")
        sys.exit(1)

    # Run the tests
    tests = [
        ("1", "Basic understanding", test_basic_understanding),
        ("2", "Quick description", test_quick_describe),
        ("3", "Timeline", test_timeline),
        ("4", "Key frames", test_key_frames),
        ("5", "Sentiment analysis", test_emotion_analysis),
        ("6", "Q&A", test_qa),
        ("7", "Custom prompt", test_custom_prompt),
        ("all", "All tests", None),
    ]

    print("\nAvailable tests:")
    for num, name, _ in tests:
        print(f"  {num}. {name}")

    choice = input("\nChoose a test to run (number or 'all'): ").strip()

    if choice == "all":
        for num, name, test_func in tests[:-1]:
            try:
                test_func()
            except Exception as e:
                print(f"\nTest {num} ({name}) failed: {e}")
    else:
        for num, name, test_func in tests:
            if choice == num and test_func:
                try:
                    test_func()
                except Exception as e:
                    print(f"\nTest failed: {e}")
                break
        else:
            print(f"Invalid choice: {choice}")

    print("\n" + "=" * 60)
    print("Tests complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
