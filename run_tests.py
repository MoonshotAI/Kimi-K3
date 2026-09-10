"""
Kimi K3 multimodal - automated tests
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
    resp = requests.get(GITHUB_RAW_URL, stream=True, timeout=120, proxies={})
    resp.raise_for_status()

    with open(LOCAL_VIDEO_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Download complete: {LOCAL_VIDEO_PATH}")


def run_image_tests():
    """Run all image tests"""
    print("\n" + "=" * 60)
    print("Image feature tests")
    print("=" * 60)

    client = create_client()

    # Test 1: image recognition
    print("\n[Test 1] Image recognition - basic description")
    try:
        result = client.describe_image(TEST_IMAGE_URL)
        print(f"OK: {result[:100]}...")
    except Exception as e:
        print(f"FAILED: {e}")

    # Test 2: object detection
    print("\n[Test 2] Image recognition - object detection")
    try:
        result = client.identify_objects(TEST_IMAGE_URL)
        print(f"OK: {result[:100]}...")
    except Exception as e:
        print(f"FAILED: {e}")

    # Test 3: scene analysis
    print("\n[Test 3] Image recognition - scene analysis")
    try:
        result = client.analyze_scene(TEST_IMAGE_URL)
        print(f"OK: {result[:100]}...")
    except Exception as e:
        print(f"FAILED: {e}")

    # Test 4: image Q&A
    print("\n[Test 4] Image recognition - Q&A mode")
    try:
        result = client.answer_image_question(TEST_IMAGE_URL, "What are the dominant colours in the image?")
        print(f"OK: {result[:100]}...")
    except Exception as e:
        print(f"FAILED: {e}")

    # Test 5: image generation
    print("\n[Test 5] Image generation")
    try:
        urls = client.generate_image("a cute ginger cat", size="1024x1024", n=1)
        print(f"OK: {urls[0][:50]}...")
    except Exception as e:
        print(f"FAILED: {e}")


def run_video_tests():
    """Run all video tests"""
    print("\n" + "=" * 60)
    print("Video feature tests")
    print("=" * 60)

    client = create_client()

    # Test 6: video understanding
    print("\n[Test 6] Video understanding - streaming output")
    try:
        print("Analyzing video...")
        count = 0
        for chunk in client.analyze_video(
            LOCAL_VIDEO_PATH,
            prompt="Describe this video in one sentence.",
            stream=True
        ):
            print(chunk, end="", flush=True)
            count += 1
            if count > 10:
                break
        print("\nOK")
    except Exception as e:
        print(f"\nFAILED: {e}")

    # Test 7: quick video description
    print("\n[Test 7] Quick video description")
    try:
        result = client.describe_video(LOCAL_VIDEO_PATH)
        print(f"OK: {result[:100]}...")
    except Exception as e:
        print(f"FAILED: {e}")

    # Test 8: video Q&A
    print("\n[Test 8] Video Q&A")
    try:
        result = client.answer_video_question(LOCAL_VIDEO_PATH, "What is in the video?")
        print(f"OK: {result[:100]}...")
    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Kimi K3 multimodal - automated tests")
    print("=" * 60)

    # Check the API key
    if not os.environ.get("KIMI_API_KEY"):
        print("\nError: set the KIMI_API_KEY environment variable first")
        sys.exit(1)

    # Download the video
    download_video()

    # Run the tests
    try:
        run_image_tests()
    except Exception as e:
        print(f"\nImage tests failed: {e}")

    try:
        run_video_tests()
    except Exception as e:
        print(f"\nVideo tests failed: {e}")

    print("\n" + "=" * 60)
    print("Tests complete.")
    print("=" * 60)
