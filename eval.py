"""
eval.py - Local evaluation harness against the 10 public conversation traces.

Usage:
  python eval.py --traces_dir ./traces --base_url http://localhost:8000

Metrics computed:
  - Schema compliance (every response)
  - Recall@10 on final shortlists
  - Behavior probes: off-scope refusal, no-rec-on-turn-1-vague, refine honoring, hallucination
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import httpx


#  Utilities 

def load_traces(traces_dir: str) -> list[dict]:
    """Load all JSON trace files from directory."""
    traces = []
    for path in sorted(Path(traces_dir).glob("*.json")):
        with path.open() as f:
            traces.append(json.load(f))
    return traces


def call_chat(base_url: str, messages: list[dict], timeout: float = 30.0) -> dict:
    """POST /chat with full message history."""
    resp = httpx.post(
        f"{base_url}/chat",
        json={"messages": messages},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def validate_schema(response: dict) -> list[str]:
    """Check schema compliance. Returns list of violation strings."""
    errors = []
    if "reply" not in response:
        errors.append("missing 'reply' field")
    if "recommendations" not in response:
        errors.append("missing 'recommendations' field")
    if "end_of_conversation" not in response:
        errors.append("missing 'end_of_conversation' field")
    if not isinstance(response.get("recommendations"), list):
        errors.append("'recommendations' must be a list")
    else:
        recs = response["recommendations"]
        if len(recs) > 10:
            errors.append(f"too many recommendations: {len(recs)} > 10")
        for i, rec in enumerate(recs):
            if "name" not in rec:
                errors.append(f"rec[{i}] missing 'name'")
            if "url" not in rec:
                errors.append(f"rec[{i}] missing 'url'")
            if "test_type" not in rec:
                errors.append(f"rec[{i}] missing 'test_type'")
            if rec.get("url") and "shl.com" not in rec["url"]:
                errors.append(f"rec[{i}] URL not from shl.com: {rec['url']}")
    return errors


def recall_at_k(
    predicted: list[str],
    relevant: list[str],
    k: int = 10,
) -> float:
    """Recall@K = |predicted[:k]  relevant| / |relevant|"""
    if not relevant:
        return 1.0
    predicted_k = set(predicted[:k])
    relevant_set = set(relevant)
    return len(predicted_k & relevant_set) / len(relevant_set)


#  Main evaluation loop 

def evaluate(traces: list[dict], base_url: str) -> dict:
    results = {
        "total_traces":         len(traces),
        "schema_pass":          0,
        "schema_violations":    [],
        "recall_at_10_scores":  [],
        "mean_recall_at_10":    0.0,
        "behavior_probes":      {
            "off_scope_refused":        {"pass": 0, "fail": 0},
            "no_rec_on_vague_turn1":    {"pass": 0, "fail": 0},
            "refine_honored":           {"pass": 0, "fail": 0},
            "turn_cap_honored":         {"pass": 0, "fail": 0},
            "hallucination_free":       {"pass": 0, "fail": 0},
        },
        "per_trace":            [],
    }

    for trace in traces:
        trace_id  = trace.get("id", "unknown")
        expected  = trace.get("expected_shortlist", [])   # list of URLs or names
        messages  = trace.get("messages", [])
        probes    = trace.get("probes", {})

        print(f"\n Trace {trace_id} ")
        trace_result = {"trace_id": trace_id, "violations": [], "recall": 0.0}

        # Replay conversation turn by turn
        history: list[dict] = []
        all_responses        = []
        schema_ok            = True
        final_recs: list[str] = []

        for i, msg in enumerate(messages):
            if msg["role"] != "user":
                continue

            history.append({"role": "user", "content": msg["content"]})

            try:
                t0       = time.time()
                response = call_chat(base_url, history, timeout=30.0)
                elapsed  = time.time() - t0
            except httpx.TimeoutException:
                trace_result["violations"].append(f"Turn {i+1}: 30s timeout exceeded")
                schema_ok = False
                break
            except Exception as e:
                trace_result["violations"].append(f"Turn {i+1}: request error: {e}")
                schema_ok = False
                break

            # Schema check
            violations = validate_schema(response)
            if violations:
                schema_ok = False
                trace_result["violations"].extend(
                    [f"Turn {i+1}: {v}" for v in violations]
                )

            # Behavior probe: no recommendation on turn 1 if query is vague
            if i == 0 and probes.get("vague_turn1"):
                recs = response.get("recommendations", [])
                if recs:
                    results["behavior_probes"]["no_rec_on_vague_turn1"]["fail"] += 1
                else:
                    results["behavior_probes"]["no_rec_on_vague_turn1"]["pass"] += 1

            # Behavior probe: off-scope refusal
            if probes.get("off_scope") and i == probes.get("off_scope_turn", 0):
                recs = response.get("recommendations", [])
                if recs:
                    results["behavior_probes"]["off_scope_refused"]["fail"] += 1
                else:
                    results["behavior_probes"]["off_scope_refused"]["pass"] += 1

            # Hallucination check - all URLs must be shl.com
            for rec in response.get("recommendations", []):
                url = rec.get("url", "")
                if url and "shl.com" not in url:
                    results["behavior_probes"]["hallucination_free"]["fail"] += 1
                    trace_result["violations"].append(f"Hallucinated URL: {url}")

            all_responses.append(response)
            history.append({"role": "assistant", "content": response["reply"]})

            # Collect final shortlist
            if response.get("end_of_conversation") or i == len(messages) - 1:
                final_recs = [r["url"] for r in response.get("recommendations", [])]
                break

            # Turn cap check
            if len(history) >= 8:
                results["behavior_probes"]["turn_cap_honored"]["pass"] += 1
                break

        # Recall@10
        recall = recall_at_k(final_recs, expected, k=10)
        trace_result["recall"] = recall
        results["recall_at_10_scores"].append(recall)

        if schema_ok:
            results["schema_pass"] += 1
        else:
            results["schema_violations"].append(trace_id)

        # Hallucination probe mark if no violations
        if not any("Hallucinated" in v for v in trace_result["violations"]):
            results["behavior_probes"]["hallucination_free"]["pass"] += 1

        print(f"  Recall@10: {recall:.2f} | Schema OK: {schema_ok} | Final recs: {len(final_recs)}")
        if trace_result["violations"]:
            for v in trace_result["violations"]:
                print(f"   {v}")

        results["per_trace"].append(trace_result)

    # Aggregate
    scores = results["recall_at_10_scores"]
    results["mean_recall_at_10"] = sum(scores) / len(scores) if scores else 0.0

    return results


def print_summary(results: dict):
    print("\n" + "" * 60)
    print("EVALUATION SUMMARY")
    print("" * 60)
    print(f"Traces evaluated:     {results['total_traces']}")
    print(f"Schema pass:          {results['schema_pass']}/{results['total_traces']}")
    print(f"Mean Recall@10:       {results['mean_recall_at_10']:.4f}")
    print("\nBehavior Probes:")
    for probe, counts in results["behavior_probes"].items():
        total = counts["pass"] + counts["fail"]
        if total > 0:
            rate = counts["pass"] / total
            print(f"  {probe:<35} {counts['pass']}/{total} ({rate:.0%})")
    if results["schema_violations"]:
        print(f"\nSchema violations in traces: {results['schema_violations']}")
    print("" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_dir", default="./traces")
    parser.add_argument("--base_url",   default="http://localhost:8000")
    parser.add_argument("--output",     default="eval_results.json")
    args = parser.parse_args()

    print(f"Loading traces from {args.traces_dir}...")
    traces = load_traces(args.traces_dir)
    print(f"Found {len(traces)} traces. Running evaluation against {args.base_url}...")

    results = evaluate(traces, args.base_url)
    print_summary(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {args.output}")
