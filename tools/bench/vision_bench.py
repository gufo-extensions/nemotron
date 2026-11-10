#!/usr/bin/env python3
"""Time image prompts on an OpenAI-compatible server.

Each request carries one image and a short question, greedy with thinking
off and a single output token, so its time is the prompt's: decoding and
encoding the image and prefilling the prompt. A unique system message opens
every request, so no prompt cache serves it. One warmup request per image,
then the median of --repeat requests.

    python3 tools/bench/vision_bench.py --port 8080 --out gufo.json IMAGE...
"""

import argparse
import base64
import json
import mimetypes
import statistics
import time
import urllib.request
import uuid
from pathlib import Path


def request(port, image, nonce):
    mime = mimetypes.guess_type(image)[0] or "image/png"
    data = base64.b64encode(Path(image).read_bytes()).decode()
    body = {
        "model": "nemotron-3.5-bolt",
        "messages": [
            {"role": "system", "content": f"Request {nonce}."},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{data}"}},
                {"type": "text", "text": "Describe this image."}]}],
        "max_tokens": 1, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.perf_counter()
    reply = json.load(urllib.request.urlopen(req, timeout=900))
    return time.perf_counter() - start, reply["usage"]["prompt_tokens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    results = []
    for image in args.images:
        request(args.port, image, uuid.uuid4().hex)
        times = []
        tokens = 0
        for _ in range(args.repeat):
            seconds, tokens = request(args.port, image, uuid.uuid4().hex)
            times.append(seconds)
        median = statistics.median(times)
        results.append({"image": Path(image).name, "prompt_tokens": tokens,
                        "seconds": times, "median_seconds": median})
        print(f"{Path(image).name}: {tokens} prompt tokens, "
              f"{median:.3f} s ({tokens / median:.0f} tokens/s)")
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
