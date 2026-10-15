#!/usr/bin/env python3
"""
Compare local chat models for the Harbor & Co. RAG chatbot.

Does NOT change the application default in rag_chatbot/config.py.
Uses the same retrieval/prompt preparation for both models, then generates
with each model under identical NUM_PREDICT / TEMPERATURE settings.

Usage (from project root, venv active, Ollama running):

  python scripts/compare_chat_models.py
  python scripts/compare_chat_models.py --models llama3.2:3b qwen2.5:0.5b --warm-runs 2

Output: data/evaluation/model_comparison.json
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
from rag_chatbot.ollama_client import chat_stream, list_models, unload_model
from rag_chatbot.pipeline import _prepare

OUT_PATH = config.PROJECT_ROOT / "data" / "evaluation" / "model_comparison.json"

# Representative mix: normal, edge, troubleshooting, pricing, multi, unsupported, ambiguous, order
BENCHMARK_CASE_IDS = [
    "refund_timing",
    "cancel_shipped",
    "quietbuds_price",
    "quietbuds_bluetooth",
    "multi_return_and_refund",
    "support_hours",
    "out_of_scope_capitals",
    "ambiguous_broken",
    "order_1024_status",
    "privacy_other_customer",
]

INVENTED_MARKERS = [
    "+91",
    "1-800",
    "call us at",
    "whatsapp",
    "guaranteed next-day",
    "lifetime warranty",
    "100% refund always",
]


def load_cases() -> list[dict]:
    data = json.loads(config.EVAL_PATH.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in data["cases"]}
    cases = []
    for cid in BENCHMARK_CASE_IDS:
        if cid in by_id:
            cases.append(by_id[cid])
    return cases


def score_answer(case: dict, answer: str, gated: bool, used_llm: bool) -> dict:
    text = (answer or "").lower()
    facts = case.get("expected_facts") or []

    if case.get("unsupported") or case.get("privacy_block"):
        refuse_ok = gated or any(
            m in text
            for m in [
                "not have enough information",
                "cannot list",
                "order id",
                "privacy",
            ]
        )
        return {
            "facts_ok": refuse_ok,
            "refuse_ok": refuse_ok,
            "invented": False,
            "relevant_enough": refuse_ok,
            "notes": "expected refusal / privacy / unsupported handling",
        }

    if case.get("type") == "ambiguous":
        clarify_ok = "?" in (answer or "") or "which" in text or "order id" in text
        return {
            "facts_ok": clarify_ok,
            "refuse_ok": False,
            "invented": False,
            "relevant_enough": clarify_ok,
            "notes": "expected clarifying question",
        }

    # Fact checks: allow flexible matching for short tokens
    fact_hits = 0
    for fact in facts:
        f = fact.lower()
        if f in text:
            fact_hits += 1
            continue
        # loosen "5 to 7 business days" → require 5 and 7
        nums = re.findall(r"\d+(?:\.\d+)?", f)
        if nums and all(n in text for n in nums):
            fact_hits += 1
            continue
        # "cannot be cancelled" → cannot + cancel
        if "cannot" in f and "cancel" in f and ("cannot" in text or "can't" in text) and "cancel" in text:
            fact_hits += 1
            continue
    facts_ok = fact_hits == len(facts) if facts else (not gated if used_llm else True)

    invented = any(m in text for m in INVENTED_MARKERS)
    # Weak relevance: answer not empty and not a pure off-topic dump
    relevant_enough = bool(answer.strip()) and not invented
    if facts and facts_ok:
        relevant_enough = True

    return {
        "facts_ok": facts_ok,
        "refuse_ok": False,
        "invented": invented,
        "relevant_enough": relevant_enough,
        "fact_hits": f"{fact_hits}/{len(facts)}" if facts else "n/a",
        "notes": "",
    }


def generate_timed(messages: list[dict], model: str) -> dict:
    """Stream generate; return answer + ttft_s + gen_s."""
    t0 = time.perf_counter()
    first = None
    pieces: list[str] = []
    for piece in chat_stream(
        messages,
        model=model,
        num_predict=config.NUM_PREDICT,
        temperature=config.TEMPERATURE,
        keep_alive=config.OLLAMA_KEEP_ALIVE,
    ):
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


def ensure_models(models: list[str]) -> None:
    names = list_models(force=True)
    for m in models:
        if not any(n.startswith(m) for n in names):
            raise SystemExit(f"Model not installed: {m}. Available: {sorted(names)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local Ollama chat models for RAG")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.2:3b", "qwen2.5:0.5b"],
        help="Chat models to compare (must already be installed)",
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=2,
        help="Warm generation repeats per LLM case (default 2)",
    )
    parser.add_argument(
        "--skip-cold",
        action="store_true",
        help="Skip cold-start measurement (faster)",
    )
    args = parser.parse_args()

    print(f"App default CHAT_MODEL (unchanged): {config.CHAT_MODEL}")
    print(f"Shared gen settings: NUM_PREDICT={config.NUM_PREDICT} TEMPERATURE={config.TEMPERATURE}")
    print(f"Models under test: {args.models}")
    ensure_models(args.models)

    cases = load_cases()
    print(f"Cases: {[c['id'] for c in cases]}")

    # Prepare once — same retrieval, gate, and prompt for every model
    prepared: dict[str, dict] = {}
    for case in cases:
        q = case["question"]
        t0 = time.perf_counter()
        prep = _prepare(q)
        prepared[case["id"]] = {
            "case": case,
            "prep": prep,
            "prepare_s": round(time.perf_counter() - t0, 3),
        }
        mode = "fixed" if prep["messages"] is None else "llm"
        print(f"  prepare {case['id']}: {mode} ({prepared[case['id']]['prepare_s']}s)")

    report: dict = {
        "app_default_chat_model": config.CHAT_MODEL,
        "shared_settings": {
            "num_predict": config.NUM_PREDICT,
            "temperature": config.TEMPERATURE,
            "max_distance": config.MAX_DISTANCE,
            "top_k": config.TOP_K,
            "embed_model": config.EMBED_MODEL,
        },
        "models": args.models,
        "warm_runs": args.warm_runs,
        "cases": [],
        "per_model_summary": {},
        "recommendation": "",
    }

    # Cold start: one LLM case per model (first generative case)
    cold_case_id = next(
        cid
        for cid, item in prepared.items()
        if item["prep"]["messages"] is not None
    )

    for model in args.models:
        model_rows = []
        print(f"\n=== Model: {model} ===")

        if not args.skip_cold:
            print(f"Cold start on {cold_case_id}...")
            for other in args.models:
                unload_model(other)
            time.sleep(1)
            item = prepared[cold_case_id]
            prep = item["prep"]
            t_e2e0 = time.perf_counter()
            gen = generate_timed(prep["messages"], model)
            e2e = time.perf_counter() - t_e2e0 + item["prepare_s"]
            quality = score_answer(item["case"], gen["answer"], False, True)
            cold_row = {
                "case_id": cold_case_id,
                "run_type": "cold",
                "prepare_s": item["prepare_s"],
                "ttft_s": gen["ttft_s"],
                "gen_s": gen["gen_s"],
                "e2e_s": round(e2e, 3),
                "answer_preview": gen["answer"][:180],
                "quality": quality,
            }
            model_rows.append(cold_row)
            print(
                f"  COLD ttft={gen['ttft_s']:.2f}s gen={gen['gen_s']:.2f}s "
                f"facts_ok={quality['facts_ok']} :: {gen['answer'][:80]!r}"
            )

        # Warm runs for all cases
        for case in cases:
            item = prepared[case["id"]]
            prep = item["prep"]
            case = item["case"]

            if prep["messages"] is None:
                # Fixed reply path — identical for all models (no generation)
                answer = prep["answer"] or ""
                quality = score_answer(case, answer, prep["gated"], False)
                row = {
                    "case_id": case["id"],
                    "run_type": "fixed_no_llm",
                    "prepare_s": item["prepare_s"],
                    "ttft_s": 0.0,
                    "gen_s": 0.0,
                    "e2e_s": item["prepare_s"],
                    "answer_preview": answer[:180],
                    "quality": quality,
                    "note": "Pipeline fixed reply; model not invoked",
                }
                model_rows.append(row)
                print(f"  FIXED {case['id']}: {answer[:70]!r}")
                continue

            warm_metrics = []
            last_answer = ""
            last_quality: dict = {}
            for i in range(args.warm_runs):
                t_e2e0 = time.perf_counter()
                gen = generate_timed(prep["messages"], model)
                # e2e ≈ prepare once + this generation (prepare already done)
                e2e = item["prepare_s"] + gen["gen_s"]
                quality = score_answer(case, gen["answer"], False, True)
                warm_metrics.append(
                    {
                        "run": i + 1,
                        "ttft_s": gen["ttft_s"],
                        "gen_s": gen["gen_s"],
                        "e2e_s": round(e2e, 3),
                        "quality": quality,
                    }
                )
                last_answer = gen["answer"]
                last_quality = quality
                print(
                    f"  WARM{i+1} {case['id']}: ttft={gen['ttft_s']:.2f}s "
                    f"gen={gen['gen_s']:.2f}s facts_ok={quality['facts_ok']} "
                    f"invented={quality['invented']}"
                )

            avg_ttft = sum(m["ttft_s"] for m in warm_metrics) / len(warm_metrics)
            avg_gen = sum(m["gen_s"] for m in warm_metrics) / len(warm_metrics)
            avg_e2e = sum(m["e2e_s"] for m in warm_metrics) / len(warm_metrics)
            model_rows.append(
                {
                    "case_id": case["id"],
                    "run_type": "warm",
                    "prepare_s": item["prepare_s"],
                    "ttft_s_avg": round(avg_ttft, 3),
                    "gen_s_avg": round(avg_gen, 3),
                    "e2e_s_avg": round(avg_e2e, 3),
                    "runs": warm_metrics,
                    "answer_preview": last_answer[:180],
                    "quality": last_quality,
                }
            )

        report["per_model_summary"][model] = summarize_model(model_rows)
        report["cases"].append({"model": model, "rows": model_rows})

    report["recommendation"] = build_recommendation(report)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print("\n" + report["recommendation"])
    print(f"\nWrote {OUT_PATH}")
    print(f"Default CHAT_MODEL still: {config.CHAT_MODEL}")
    return 0


def summarize_model(rows: list[dict]) -> dict:
    warm = [r for r in rows if r.get("run_type") == "warm"]
    cold = [r for r in rows if r.get("run_type") == "cold"]
    fixed = [r for r in rows if r.get("run_type") == "fixed_no_llm"]

    def avg(key: str, items: list[dict]) -> float | None:
        if not items:
            return None
        return round(sum(i[key] for i in items) / len(items), 3)

    # Quality over warm LLM rows (last quality) + fixed
    quality_rows = warm + fixed
    n = len(quality_rows) or 1
    facts_ok = sum(1 for r in quality_rows if r.get("quality", {}).get("facts_ok"))
    invented = sum(1 for r in quality_rows if r.get("quality", {}).get("invented"))
    relevant = sum(1 for r in quality_rows if r.get("quality", {}).get("relevant_enough"))

    return {
        "warm_avg_ttft_s": avg("ttft_s_avg", warm),
        "warm_avg_gen_s": avg("gen_s_avg", warm),
        "warm_avg_e2e_s": avg("e2e_s_avg", warm),
        "cold_ttft_s": cold[0]["ttft_s"] if cold else None,
        "cold_gen_s": cold[0]["gen_s"] if cold else None,
        "fixed_reply_cases": len(fixed),
        "llm_warm_cases": len(warm),
        "facts_ok_rate": round(facts_ok / n, 3),
        "invented_count": invented,
        "relevant_rate": round(relevant / n, 3),
    }


def build_recommendation(report: dict) -> str:
    summaries = report["per_model_summary"]
    if len(summaries) < 2:
        return "Need at least two models to recommend."

    lines = ["COMPARISON SUMMARY"]
    for model, s in summaries.items():
        lines.append(
            f"- {model}: warm_gen_avg={s['warm_avg_gen_s']}s "
            f"warm_ttft_avg={s['warm_avg_ttft_s']}s "
            f"facts_ok_rate={s['facts_ok_rate']} "
            f"invented={s['invented_count']} "
            f"cold_gen={s['cold_gen_s']}s"
        )

    # Rank: prefer higher facts_ok, then lower invented, then speed
    ranked = sorted(
        summaries.items(),
        key=lambda kv: (
            -(kv[1]["facts_ok_rate"] or 0),
            kv[1]["invented_count"],
            kv[1]["warm_avg_gen_s"] or 9999,
        ),
    )
    best, best_s = ranked[0]
    fastest = min(summaries.items(), key=lambda kv: kv[1]["warm_avg_gen_s"] or 9999)

    lines.append("")
    lines.append(
        f"Quality-first pick (this run): {best} "
        f"(facts_ok_rate={best_s['facts_ok_rate']}, invented={best_s['invented_count']})."
    )
    lines.append(
        f"Speed-first pick (this run): {fastest[0]} "
        f"(warm_gen_avg={fastest[1]['warm_avg_gen_s']}s)."
    )
    lines.append(
        "Trade-off: smaller models answer faster but may miss policy nuances or invent details; "
        "larger models are slower on CPU but often more reliable for support grounding."
    )
    lines.append(
        "Default CHAT_MODEL was NOT changed. Review results before switching production default."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
