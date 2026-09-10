"""
Kimi K3 multimodal - full feature tests
Tests image recognition, image generation, video understanding, and video generation.
"""

import os
import sys
from kimi_multimodal import KimiClient, create_client

# === Configuration ===
GITHUB_RAW_URL = "https://github.com/CherrieTheShifter/Kimi-K3/raw/main/orange-cat-agent.mp4"
LOCAL_VIDEO_PATH = "./orange-cat-agent.mp4"
TEST_IMAGE_URL = "https://picsum.photos/800/600"


def download_video():
    """Download the test video"""
    import requests

    if os.path.exists(LOCAL_VIDEO_PATH):
        print(f"[setup] Video already present, skipping download")
        return

    print(f"[setup] Downloading video from GitHub...")
    resp = requests.get(GITHUB_RAW_URL, stream=True, timeout=120)
    resp.raise_for_status()

    with open(LOCAL_VIDEO_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Download complete: {LOCAL_VIDEO_PATH}")


# ============================================================
# Image recognition tests
# ============================================================

def test_image_recognition():
    """Test image recognition"""
    print("\n" + "=" * 60)
    print("Test 1: image recognition - basic description")
    print("=" * 60)

    client = create_client()
    result = client.describe_image(TEST_IMAGE_URL)
    print(f"\n{result}")


def test_image_objects():
    """Test object detection"""
    print("\n" + "=" * 60)
    print("Test 2: image recognition - object detection")
    print("=" * 60)

    client = create_client()
    result = client.identify_objects(TEST_IMAGE_URL)
    print(f"\n{result}")


def test_image_scene():
    """Test scene analysis"""
    print("\n" + "=" * 60)
    print("Test 3: image recognition - scene analysis")
    print("=" * 60)

    client = create_client()
    result = client.analyze_scene(TEST_IMAGE_URL)
    print(f"\n{result}")


def test_image_qa():
    """Test image Q&A"""
    print("\n" + "=" * 60)
    print("Test 4: image recognition - Q&A mode")
    print("=" * 60)

    client = create_client()
    questions = [
        "What are the dominant colours in the image?",
        "What objects are in the image?",
        "What feeling does this image give?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer = client.answer_image_question(TEST_IMAGE_URL, q)
        print(f"Answer: {answer}")


# ============================================================
# Image generation tests
# ============================================================

def test_image_generation():
    """Test image generation"""
    print("\n" + "=" * 60)
    print("Test 5: image generation")
    print("=" * 60)

    client = create_client()

    prompt = "a cute ginger cat dozing in the sunshine, cosy scene, high-definition photography"
    print(f"Prompt: {prompt}")

    urls = client.generate_image(prompt, size="1024x1024", n=1)
    print(f"\nGenerated successfully.")
    for i, url in enumerate(urls, 1):
        print(f"  Image {i}: {url}")


def test_image_generation_and_save():
    """Test image generation and saving"""
    print("\n" + "=" * 60)
    print("Test 6: generate an image and save it locally")
    print("=" * 60)

    client = create_client()

    prompt = "a futuristic city at night, flickering neon, sci-fi style"
    output_path = "./generated_image.png"

    print(f"Prompt: {prompt}")
    client.generate_and_save_image(prompt, output_path, size="1024x1024")
    print(f"Image saved: {output_path}")


def test_image_styles():
    """Test image generation across styles"""
    print("\n" + "=" * 60)
    print("Test 7: image generation in different styles")
    print("=" * 60)

    client = create_client()

    styles = [
        ("Photorealistic", "a ginger cat, realistic photo style"),
        ("Cartoon/anime", "a ginger cat, cartoon anime style"),
        ("Oil painting", "a ginger cat, Van Gogh oil painting style"),
    ]

    for style_name, prompt in styles:
        print(f"\nStyle: {style_name}")
        urls = client.generate_image(prompt, size="1024x1024")
        print(f"  Generated: {urls[0][:50]}...")


# ============================================================
# Video understanding tests
# ============================================================

def test_video_understanding():
    """Test video understanding"""
    print("\n" + "=" * 60)
    print("Test 8: video understanding - streaming output")
    print("=" * 60)

    client = create_client()

    print("\nAnalyzing video...")
    for chunk in client.analyze_video(
        LOCAL_VIDEO_PATH,
        prompt="Describe the content of this video in detail.",
        stream=True
    ):
        print(chunk, end="", flush=True)
    print()


def test_video_describe():
    """Test quick video description"""
    print("\n" + "=" * 60)
    print("Test 9: quick video description")
    print("=" * 60)

    client = create_client()
    result = client.describe_video(LOCAL_VIDEO_PATH)
    print(f"\n{result}")


def test_video_timeline():
    """Test video timeline analysis"""
    print("\n" + "=" * 60)
    print("Test 10: video timeline analysis")
    print("=" * 60)

    client = create_client()
    result = client.analyze_timeline(LOCAL_VIDEO_PATH)
    print(f"\n{result}")


def test_video_qa():
    """Test video Q&A"""
    print("\n" + "=" * 60)
    print("Test 11: video Q&A")
    print("=" * 60)

    client = create_client()
    questions = [
        "How many people are in the video?",
        "What kind of background music does it use?",
        "Was it filmed indoors or outdoors?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer = client.answer_video_question(LOCAL_VIDEO_PATH, q)
        print(f"Answer: {answer}")


# ============================================================
# Video generation tests
# ============================================================

def test_video_generation():
    """Test video generation"""
    print("\n" + "=" * 60)
    print("Test 12: video generation")
    print("=" * 60)

    client = create_client()

    prompt = "a ginger cat playing on the grass, bright sunshine, warm and cute"
    print(f"Prompt: {prompt}")

    url = client.generate_video(prompt, duration=5, size="1280x720")
    print(f"\nGenerated successfully.")
    print(f"  Video URL: {url}")


def test_video_generation_and_save():
    """Test video generation and saving"""
    print("\n" + "=" * 60)
    print("Test 13: generate a video and save it locally")
    print("=" * 60)

    client = create_client()

    prompt = "a beach at sunset, waves lapping gently at the sand"
    output_path = "./generated_video.mp4"

    print(f"Prompt: {prompt}")
    client.generate_and_save_video(prompt, output_path, duration=5)
    print(f"Video saved: {output_path}")


# ============================================================
# Main flow
# ============================================================

def main():
    """Main test flow"""
    print("=" * 60)
    print("Kimi K3 multimodal - full feature tests")
    print("=" * 60)

    # Check the API key
    if not os.environ.get("KIMI_API_KEY"):
        print("\nError: set the KIMI_API_KEY environment variable first")
        print("  export KIMI_API_KEY=\"your_api_key\"")
        print("  Get an API key at https://platform.kimi.ai")
        sys.exit(1)

    # Download the video
    download_video()

    # Test menu
    tests = [
        ("1", "Image recognition - basic", test_image_recognition),
        ("2", "Image recognition - objects", test_image_objects),
        ("3", "Image recognition - scene", test_image_scene),
        ("4", "Image recognition - Q&A", test_image_qa),
        ("5", "Image generation", test_image_generation),
        ("6", "Image generation and save", test_image_generation_and_save),
        ("7", "Image generation - styles", test_image_styles),
        ("8", "Video understanding - streaming", test_video_understanding),
        ("9", "Quick video description", test_video_describe),
        ("10", "Video timeline analysis", test_video_timeline),
        ("11", "Video Q&A", test_video_qa),
        ("12", "Video generation", test_video_generation),
        ("13", "Video generation and save", test_video_generation_and_save),
        ("img", "All image tests", None),
        ("vid", "All video tests", None),
        ("all", "All tests", None),
    ]

    print("\nAvailable tests:")
    for num, name, _ in tests:
        print(f"  {num}. {name}")

    choice = input("\nChoose a test to run (number or 'all'): ").strip()

    if choice == "all":
        for num, name, test_func in tests[:-3]:
            if test_func:
                try:
                    test_func()
                except Exception as e:
                    print(f"\nTest {num} ({name}) failed: {e}")
    elif choice == "img":
        for test_func in [test_image_recognition, test_image_objects, test_image_scene, test_image_qa,
                         test_image_generation, test_image_generation_and_save, test_image_styles]:
            try:
                test_func()
            except Exception as e:
                print(f"\nImage tests failed: {e}")
    elif choice == "vid":
        for test_func in [test_video_understanding, test_video_describe, test_video_timeline, test_video_qa,
                         test_video_generation, test_video_generation_and_save]:
            try:
                test_func()
            except Exception as e:
                print(f"\nVideo tests failed: {e}")
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
