"""
Command-line interface for the Kimi K3 client.

    kimi-k3 chat "hello"
    kimi-k3 image ./cat.png --prompt "describe this"
    kimi-k3 video ./clip.mp4 --timeline
    kimi-k3 gen-image "a ginger cat" -o out.png
    kimi-k3 gen-video "waves at sunset" -o out.mp4

Requires KIMI_API_KEY in the environment. Run any subcommand with --help
for its options.
"""

import argparse
import sys
from typing import Optional

from kimi_multimodal import KimiClient, VALID_REASONING_EFFORTS, DEFAULT_REASONING_EFFORT


def _client(args) -> KimiClient:
    return KimiClient(model=args.model, max_retries=args.max_retries)


def _emit(result, stream: bool) -> None:
    """Print a result, handling both plain strings and (kind, text) generators."""
    if not stream:
        print(result)
        return
    for chunk in result:
        text = chunk[1] if isinstance(chunk, tuple) else chunk
        print(text, end="", flush=True)
    print()


def _show_usage(client: KimiClient) -> None:
    u = client.last_usage
    if not u:
        return
    print(
        f"\n[tokens] prompt={u.get('prompt_tokens', 0)} "
        f"completion={u.get('completion_tokens', 0)} "
        f"reasoning={u.get('reasoning_tokens', 0)} "
        f"total={u.get('total_tokens', 0)}",
        file=sys.stderr,
    )


def cmd_chat(args) -> int:
    client = _client(args)
    result = client.chat(args.prompt, system=args.system, stream=args.stream,
                         reasoning_effort=args.effort)
    _emit(result, args.stream)
    if args.usage:
        _show_usage(client)
    return 0


def cmd_image(args) -> int:
    client = _client(args)
    if args.ocr:
        result = client.extract_text_from_image(args.source)
    elif args.objects:
        result = client.identify_objects(args.source)
    elif args.scene:
        result = client.analyze_scene(args.source)
    elif args.ask:
        result = client.answer_image_question(args.source, args.ask)
    else:
        result = client.analyze_image(args.source, prompt=args.prompt,
                                      stream=args.stream, reasoning_effort=args.effort)
    _emit(result, args.stream and not (args.ocr or args.objects or args.scene or args.ask))
    if args.usage:
        _show_usage(client)
    return 0


def cmd_video(args) -> int:
    client = _client(args)
    if args.timeline:
        result = client.analyze_timeline(args.source)
    elif args.key_frames:
        result = client.extract_key_frames(args.source)
    elif args.emotion:
        result = client.analyze_emotion(args.source)
    elif args.ask:
        result = client.answer_video_question(args.source, args.ask)
    else:
        result = client.analyze_video(args.source, prompt=args.prompt,
                                      stream=args.stream, reasoning_effort=args.effort)
        _emit(result, args.stream)
        if args.usage:
            _show_usage(client)
        return 0
    print(result)
    if args.usage:
        _show_usage(client)
    return 0


def cmd_gen_image(args) -> int:
    client = _client(args)
    if args.output:
        path = client.generate_and_save_image(args.prompt, args.output,
                                              size=args.size, style=args.style)
        print(path)
    else:
        for url in client.generate_image(args.prompt, size=args.size,
                                         n=args.n, style=args.style):
            print(url)
    return 0


def cmd_gen_video(args) -> int:
    client = _client(args)
    if args.output:
        print(client.generate_and_save_video(args.prompt, args.output,
                                             duration=args.duration, size=args.size))
    else:
        print(client.generate_video(args.prompt, duration=args.duration, size=args.size))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kimi-k3",
        description="Command-line client for the Kimi K3 API.",
    )
    p.add_argument("--model", default="kimi-k3", help="model name (default: kimi-k3)")
    p.add_argument("--effort", default=DEFAULT_REASONING_EFFORT,
                   choices=VALID_REASONING_EFFORTS, help="reasoning effort")
    p.add_argument("--max-retries", type=int, default=3,
                   help="retry attempts on 429 and 5xx (default: 3)")
    p.add_argument("--usage", action="store_true",
                   help="print token usage to stderr after the call")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("chat", help="text chat")
    c.add_argument("prompt")
    c.add_argument("--system", help="system prompt")
    c.add_argument("--stream", action="store_true")
    c.set_defaults(func=cmd_chat)

    i = sub.add_parser("image", help="analyse an image")
    i.add_argument("source", help="local path, http(s) URL, or base64: prefix")
    i.add_argument("--prompt", default="Describe this image in detail.")
    i.add_argument("--ocr", action="store_true", help="extract text")
    i.add_argument("--objects", action="store_true", help="detect objects")
    i.add_argument("--scene", action="store_true", help="analyse the scene")
    i.add_argument("--ask", help="ask a specific question about the image")
    i.add_argument("--stream", action="store_true")
    i.set_defaults(func=cmd_image)

    v = sub.add_parser("video", help="analyse a video")
    v.add_argument("source", help="local path, http(s) URL, or base64: prefix")
    v.add_argument("--prompt", default="Describe this video in detail.")
    v.add_argument("--timeline", action="store_true", help="chronological breakdown")
    v.add_argument("--key-frames", action="store_true", help="key frame descriptions")
    v.add_argument("--emotion", action="store_true", help="mood and atmosphere")
    v.add_argument("--ask", help="ask a specific question about the video")
    v.add_argument("--stream", action="store_true")
    v.set_defaults(func=cmd_video)

    gi = sub.add_parser("gen-image", help="generate an image")
    gi.add_argument("prompt")
    gi.add_argument("-o", "--output", help="save to this path instead of printing URLs")
    gi.add_argument("--size", default="1024x1024")
    gi.add_argument("-n", type=int, default=1, help="how many to generate")
    gi.add_argument("--style")
    gi.set_defaults(func=cmd_gen_image)

    gv = sub.add_parser("gen-video", help="generate a video")
    gv.add_argument("prompt")
    gv.add_argument("-o", "--output", help="save to this path instead of printing the URL")
    gv.add_argument("--size", default="1280x720")
    gv.add_argument("--duration", type=int, default=5, help="seconds")
    gv.set_defaults(func=cmd_gen_video)

    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:          # missing API key
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
