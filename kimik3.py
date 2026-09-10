"""
Kimi K3 - preserved thinking history example.

K3 is trained in preserved-thinking-history mode. For multi-turn conversations
and tool calls, the complete assistant message returned by the API must be passed
back to `messages` as-is - including `reasoning_content` and `tool_calls`, not
just `content`. Dropping the reasoning loses context between turns.

Run:
    export KIMI_API_KEY="your_api_key"
    python kimik3.py
"""

import os

import openai


MODEL = "kimi-k3"
BASE_URL = "https://api.moonshot.cn/v1"


def chat_with_preserved_thinking(client: openai.OpenAI, model_name: str) -> str:
    """
    Ask a follow-up that can only be answered from the previous turn's reasoning.

    The assistant should mention 215 and 222 - numbers that appear only in the
    prior `reasoning_content`, never in the visible answer.
    """
    messages = [
        {
            "role": "user",
            "content": "Tell me three random numbers.",
        },
        {
            "role": "assistant",
            "reasoning_content": (
                "I'll start by listing five numbers: 473, 921, 235, 215, 222, "
                "and I'll tell you the first three."
            ),
            "content": "473, 921, 235",
        },
        {
            "role": "user",
            "content": "What are the other two numbers you have in mind?",
        },
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        max_tokens=4096,
        reasoning_effort="max",
    )

    message = response.choices[0].message
    print(f"reasoning: {message.reasoning_content}")
    return message.content


def main() -> None:
    api_key = os.environ.get("KIMI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Error: set the KIMI_API_KEY environment variable first\n"
            '  export KIMI_API_KEY="your_api_key"\n'
            "  Get an API key at https://platform.kimi.ai"
        )

    client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)
    print(chat_with_preserved_thinking(client, MODEL))


if __name__ == "__main__":
    main()
