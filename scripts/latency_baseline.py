#!/usr/bin/env python3
"""
Reproducible latency + correctness baseline for the Harbor & Co. RAG chatbot.

Uses a small representative eval subset. Keeps the app default CHAT_MODEL.
Writes JSON under data/evaluation/ (does not change production config).

Usage:
  python scripts/latency_baseline.py --label before --warm-runs 2
  python scripts/latency_baseline.py --label after --warm-runs 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_chatbot import config
from rag_chatbot.ollama_client import chat_stream, unload_model
from rag_chatbot.pipeline import _prepare

CASE_IDS = [
    "refund_timing",
    "cancel_shipped",
    "multi_return_and_refund",
    "support_hours",
    "out_of_scope_capitals",
    "privacy_other_customer",
]

# Flexible fact checks (cancellation allows paraphrases)
FACT_CHECKS = {
    "refund_timing": [r"5\s*(to|-)?\s*7"],
    "cancel_shipped": [
        r"cannot\s+be\s+cancell|can'?t\s+cancell|cannot\s+cancell|"
        r"won'?t\s+be\s+able\s+to\s+cancell|not\s+(be\s+)?cancell|"
        r"not\s+eligible\s+for\s+cancell|ineligible\s+for\s+cancell"
    ],
    "multi_return_and_refund": [r"damag|defect|return", r"refund"],
    "support_hours": [r"9\s*:?\s*00|9\s*am", r"18\s*:?\s*00|6\s*:?\s*00\s*pm|6\s*pm", r"\bist\b"],
}


def load_cases() -> list[dict]:
    data = json.loads(config.EVAL_PATH.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in data["cases"]}
    return [by_id[i] for i in CASE_IDS if i in by_id]


def score(case: dict, answer: str, gated: bool, used_llm: bool) -> dict:
    text = (answer or "").lower()
    cid = case["id"]

    if case.get("unsupported") or case.get("privacy_block"):
        refuse_ok = gated or any(
            m in text
            for m in ("not have enough information", "cannot list", "order id", "privacy")
        )
        return {"ok": refuse_ok, "kind": "refuse", "detail": "refusal/privacy"}

    patterns = FACT_CHECKS.get(cid) or []
    if patterns:
        hits = sum(1 for p in patterns if re.search(p, text, re.I))
        return {
            "ok": hits == len(patterns),
            "kind": "facts",
            "detail": f"{hits}/{len(patterns)}",
        }

    return {"ok": bool(answer.strip()), "kind": "nonempty", "detail": ""}


def generate_timed(messages: list[dict]) -> dict:
    t0 = time.perf_counter()
    first = None
    pieces: list[str] = []
    for piece in chat_stream(messages):
        if first is None:
            first = time.perf_counter()
        pieces.append(piece)
    t1 = time.perf_counter()
    answer = "".join(pieces).strip()
    return {
        "answer": answer,
        "ttft_s": round((first - t0) if first else (t1 - t0), 3),
        "gen_s": round(t1 - t0, 3),
        "chars": len(answer),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="Output label: before|after|…")
    parser.add_argument("--warm-runs", type=int, default=2)
    parser.add_argument("--skip-cold", action="store_true")
    args = parser.parse_args()

    out = (
        config.PROJECT_ROOT
        / "data"
        / "evaluation"
        / f"latency_{args.label}.json"
    )

    cases = load_cases()
    print(f"CHAT_MODEL={config.CHAT_MODEL} NUM_PREDICT={config.NUM_PREDICT}")
    print(f"TOP_K={config.TOP_K} MAX_DISTANCE={config.MAX_DISTANCE}")
    print(f"Cases: {[c['id'] for c in cases]}")

    prepared = {}
    for case in cases:
        t0 = time.perf_counter()
        prep = _prepare(case["question"])
        prepared[case["id"]] = {
            "case": case,
            "prep": prep,
            "prepare_s": round(time.perf_counter() - t0, 3),
        }
        mode = "fixed" if prep["messages"] is None else "llm"
        hits = [h["source"] for h in (prep.get("hits") or [])[:4]]
        print(f"  prepare {case['id']}: {mode} {prepared[case['id']]['prepare_s']}s hits={hits}")

    report: dict = {
        "label": args.label,
        "chat_model": config.CHAT_MODEL,
        "settings": {
            "num_predict": config.NUM_PREDICT,
            "temperature": config.TEMPERATURE,
            "top_k": config.TOP_K,
            "max_distance": config.MAX_DISTANCE,
            "stream_responses": config.STREAM_RESPONSES,
        },
        "warm_runs": args.warm_runs,
        "rows": [],
        "summary": {},
    }

    cold_id = next(
        cid
        for cid, item in prepared.items()
        if item["prep"]["messages"] is not None
    )

    if not args.skip_cold:
        print(f"Cold start on {cold_id}…")
        unload_model(config.CHAT_MODEL)
        time.sleep(1)
        item = prepared[cold_id]
        gen = generate_timed(item["prep"]["messages"])
        quality = score(item["case"], gen["answer"], False, True)
        report["rows"].append(
            {
                "case_id": cold_id,
                "run_type": "cold",
                "ttft_s": gen["ttft_s"],
                "gen_s": gen["gen_s"],
                "prepare_s": item["prepare_s"],
                "e2e_s": round(item["prepare_s"] + gen["gen_s"], 3),
                "quality": quality,
                "answer_preview": gen["answer"][:160],
                "prompt_chars": sum(len(m["content"]) for m in item["prep"]["messages"]),
            }
        )
        print(
            f"  COLD ttft={gen['ttft_s']:.1f}s gen={gen['gen_s']:.1f}s ok={quality['ok']}"
        )

    warm_ttft: list[float] = []
    warm_gen: list[float] = []
    warm_ttft_steady: list[float] = []
    warm_gen_steady: list[float] = []
    quality_ok = 0
    quality_n = 0

    for case in cases:
        item = prepared[case["id"]]
        prep = item["prep"]
        case = item["case"]

        if prep["messages"] is None:
            answer = prep["answer"] or ""
            quality = score(case, answer, prep["gated"], False)
            quality_n += 1
            quality_ok += int(quality["ok"])
            report["rows"].append(
                {
                    "case_id": case["id"],
                    "run_type": "fixed_no_llm",
                    "ttft_s": 0.0,
                    "gen_s": 0.0,
                    "prepare_s": item["prepare_s"],
                    "e2e_s": item["prepare_s"],
                    "quality": quality,
                    "answer_preview": answer[:160],
                }
            )
            print(f"  FIXED {case['id']}: ok={quality['ok']}")
            continue

        runs = []
        last_answer = ""
        last_q: dict = {}
        for i in range(args.warm_runs):
            gen = generate_timed(prep["messages"])
            quality = score(case, gen["answer"], False, True)
            runs.append(
                {
                    "run": i + 1,
                    "ttft_s": gen["ttft_s"],
                    "gen_s": gen["gen_s"],
                    "quality": quality,
                    "chars": gen["chars"],
                }
            )
            warm_ttft.append(gen["ttft_s"])
            warm_gen.append(gen["gen_s"])
            if i + 1 == args.warm_runs:
                warm_ttft_steady.append(gen["ttft_s"])
                warm_gen_steady.append(gen["gen_s"])
            last_answer = gen["answer"]
            last_q = quality
            print(
                f"  WARM{i+1} {case['id']}: ttft={gen['ttft_s']:.1f}s "
                f"gen={gen['gen_s']:.1f}s ok={quality['ok']} {quality.get('detail','')}"
            )

        quality_n += 1
        quality_ok += int(last_q["ok"])
        report["rows"].append(
            {
                "case_id": case["id"],
                "run_type": "warm",
                "prepare_s": item["prepare_s"],
                "ttft_s_avg": round(sum(r["ttft_s"] for r in runs) / len(runs), 3),
                "gen_s_avg": round(sum(r["gen_s"] for r in runs) / len(runs), 3),
                "ttft_s_steady": runs[-1]["ttft_s"],
                "gen_s_steady": runs[-1]["gen_s"],
                "runs": runs,
                "quality": last_q,
                "answer_preview": last_answer[:160],
                "prompt_chars": sum(len(m["content"]) for m in prep["messages"]),
                "hit_sources": [h["source"] for h in (prep.get("hits") or [])[:4]],
            }
        )

    report["summary"] = {
        "warm_avg_ttft_s": round(sum(warm_ttft) / len(warm_ttft), 3) if warm_ttft else None,
        "warm_avg_gen_s": round(sum(warm_gen) / len(warm_gen), 3) if warm_gen else None,
        "steady_avg_ttft_s": round(sum(warm_ttft_steady) / len(warm_ttft_steady), 3)
        if warm_ttft_steady
        else None,
        "steady_avg_gen_s": round(sum(warm_gen_steady) / len(warm_gen_steady), 3)
        if warm_gen_steady
        else None,
        "correctness_rate": round(quality_ok / quality_n, 3) if quality_n else None,
        "correctness": f"{quality_ok}/{quality_n}",
    }
    cold = next((r for r in report["rows"] if r["run_type"] == "cold"), None)
    if cold:
        report["summary"]["cold_ttft_s"] = cold["ttft_s"]
        report["summary"]["cold_gen_s"] = cold["gen_s"]

    out.write_text(json.dumps(report, indent=2))
    print("\nSUMMARY:", json.dumps(report["summary"], indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
