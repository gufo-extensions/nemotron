#!/usr/bin/env python3
"""Measure how long a streaming request stalls beside a cold prompt.

Request A streams a long greedy answer. Once it has sent 40 chunks, request
B sends a cold prompt and asks for one output token: either about --words
words of text, or --image with "Describe this image.". The script reports
B's time and A's longest gaps between chunks while B was in flight, with the
start of each gap relative to B's request.

    python3 tools/bench/decode_stalls.py --port 8080 --out stalls.jsonl \
        --label gufo --runs 3 [--image screen.png | --words 4500]
"""

import argparse
import base64
import http.client
import json
import random
import sys
import threading
import time

VOCAB = ("river stone harbor lantern copper meadow signal winter orchard "
         "valley engine market garden thunder ladder candle voyage silver "
         "forest bridge circuit pebble canyon ember").split()


def model_id(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    conn.request("GET", "/v1/models")
    models = json.loads(conn.getresponse().read())["data"]
    conn.close()
    return models[0]["id"]


def post(port, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        sys.exit(f"HTTP {resp.status}: {resp.read()[:500]!r}")
    return conn, resp


def decoder(port, model, state):
    try:
        body = {"model": model, "temperature": 0, "max_tokens": 3000,
                "stream": True, "reasoning_effort": "none",
                "messages": [{"role": "user", "content":
                              "Write a long, detailed essay about the history "
                              "of lighthouses, their keepers and their "
                              "technology."}]}
        conn, resp = post(port, body)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            for choice in chunk.get("choices", []):
                if choice.get("delta", {}).get("content"):
                    state["times"].append(time.monotonic())
            if state["stop"].is_set():
                break
        conn.close()
    except BaseException as error:  # noqa: BLE001
        state["error"] = repr(error)
        state["stop"].set()


def cold_request(model, args, nonce):
    if args.image:
        with open(args.image, "rb") as f:
            url = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        messages = [
            {"role": "system", "content": f"Session {nonce}."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "Describe this image."}]}]
    else:
        rng = random.Random(nonce)
        text = " ".join(rng.choice(VOCAB) for _ in range(args.words))
        messages = [{"role": "user", "content":
                     f"Session {nonce}. Summarize these notes in one word: {text}"}]
    return {"model": model, "temperature": 0, "max_tokens": 1, "stream": False,
            "reasoning_effort": "none", "messages": messages}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--words", type=int, default=4500)
    parser.add_argument("--image")
    args = parser.parse_args()
    model = model_id(args.port)
    rows = []
    for run in range(args.runs):
        state = {"times": [], "stop": threading.Event(), "error": None}
        thread = threading.Thread(target=decoder, args=(args.port, model, state),
                                  daemon=True)
        thread.start()
        while len(state["times"]) < 40:
            if state["error"]:
                sys.exit(f"decoder failed: {state['error']}")
            time.sleep(0.01)
        time.sleep(0.5)
        body = cold_request(model, args, f"{args.label}-{run}-{random.random():.12f}")
        start = time.monotonic()
        conn, resp = post(args.port, body)
        reply = json.loads(resp.read())
        end = time.monotonic()
        conn.close()
        time.sleep(1.0)
        state["stop"].set()
        thread.join(timeout=60)
        times = state["times"]
        gaps = sorted(((t2 - t1, t1 - start) for t1, t2 in zip(times, times[1:])
                       if t2 > start and t1 < end), reverse=True)
        usage = reply.get("usage", {})
        gufo = usage.get("gufo", {})
        row = {
            "label": args.label, "run": run,
            "kind": "image" if args.image else f"text{args.words}",
            "b_seconds": round(end - start, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "a_chunks_during": sum(1 for t in times if start <= t <= end),
            "a_max_gap_ms": round(1000 * gaps[0][0], 1) if gaps else None,
            "a_gaps_over_300ms": sum(1 for gap, _ in gaps if gap > 0.3),
            # Longest gaps as [gap ms, start ms relative to B's request].
            "a_top_gaps": [[round(1000 * gap), round(1000 * at)]
                           for gap, at in gaps[:4]],
            "max_prefill_layer_groups": gufo.get("max_prefill_layer_groups"),
            "prefill_yield_rounds": gufo.get("prefill_yield_rounds"),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        time.sleep(1.0)
    with open(args.out, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
