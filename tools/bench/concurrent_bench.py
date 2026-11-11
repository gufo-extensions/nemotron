#!/usr/bin/env python3
"""Measure concurrent greedy requests on an OpenAI-compatible server.

All requests of a mix start together, thinking off, at temperature 0. For
each request the script reports its decode rate, its drafting statistics, the
widest batch it decoded in and a hash of its text (to compare outputs across
builds); for the mix, the tokens of all requests per second of wall time.

    python3 tools/bench/concurrent_bench.py --port 8080 --out conc.jsonl \
        --label gufo --tokens 384 --runs 2 code,code2 prose,tcp
"""

import argparse
import hashlib
import http.client
import json
import sys
import threading
import time

PROMPTS = {
    "code": "Write a Python function that returns the n-th Fibonacci number "
            "iteratively, with a docstring and two doctests.",
    "code2": "Write a Python class for a fixed-size ring buffer with push, pop, "
             "len and iteration, with type hints and docstrings.",
    "prose": "Write a short essay about the history of the lighthouse keepers on "
             "the Atlantic coast and how their work changed over two centuries.",
    "json": "Write a JSON schema for a library book record with title, authors, "
            "ISBN, publication year, and availability, then an example record.",
    "tcp": "Explain how TCP congestion control works, covering slow start, "
           "congestion avoidance, fast retransmit and fast recovery.",
}


def model_id(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    conn.request("GET", "/v1/models")
    models = json.loads(conn.getresponse().read())["data"]
    conn.close()
    return models[0]["id"]


def one(port, model, tokens, kind, slot, results):
    body = {"model": model, "temperature": 0, "max_tokens": tokens,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": PROMPTS[kind]}]}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    start = time.monotonic()
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    reply = json.loads(resp.read())
    end = time.monotonic()
    conn.close()
    if resp.status != 200:
        results[slot] = {"kind": kind, "error": f"HTTP {resp.status}: {str(reply)[:300]}"}
        return
    usage = reply.get("usage", {})
    gufo = usage.get("gufo", {})
    text = reply["choices"][0]["message"].get("content") or ""
    results[slot] = {
        "kind": kind, "seconds": round(end - start, 3),
        "completion_tokens": usage.get("completion_tokens"),
        "tps": usage.get("completion_tokens_per_second"),
        "draft_tokens": usage.get("draft_tokens"),
        "draft_accepted": usage.get("draft_tokens_accepted"),
        "draft_rounds": gufo.get("draft_rounds"),
        "width": gufo.get("physical_execution_width"),
        "text_md5": hashlib.md5(text.encode()).hexdigest()[:10],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("mixes", nargs="+",
                        help="comma-separated kinds: " + ", ".join(PROMPTS))
    args = parser.parse_args()
    model = model_id(args.port)
    rows = []
    for mix in args.mixes:
        kinds = mix.split(",")
        for run in range(args.runs):
            results = [None] * len(kinds)
            threads = [threading.Thread(
                target=one, args=(args.port, model, args.tokens, kind, i, results),
                daemon=True) for i, kind in enumerate(kinds)]
            start = time.monotonic()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            wall = time.monotonic() - start
            failed = [r for r in results if r is None or "error" in r]
            if failed:
                sys.exit(f"request failed: {failed}")
            total = sum(r["completion_tokens"] or 0 for r in results)
            row = {"label": args.label, "mix": mix, "run": run,
                   "wall": round(wall, 3), "aggregate_tps": round(total / wall, 2),
                   "requests": results}
            rows.append(row)
            print(json.dumps(row), flush=True)
            time.sleep(1.0)
    with open(args.out, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
