"""
Hybrid evaluation: rule-based checks + LLM-as-judge (Groq)

Usage:
  python eval.py                                          # local
  python eval.py --url https://render-url.onrender.com    # deployed version
  python eval.py --skip-judge   # run only rule-based checks
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
import httpx
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

# Test cases for valid ques and guardrail checks

TEST_CASES = [
    # Valid analytics questions
    {
        "question": "What is the overall cancellation rate?",
        "expect": "answer",
        "category": "Basic metric",
        "note": "Single aggregate - IS_CANCELED rate across all bookings",
    },
    {
        "question": "Which market segment has the highest cancellation rate?",
        "expect": "answer",
        "category": "Segment analysis",
        "note": "Core Expedia question - Online TA should be highest",
    },
    {
        "question": "What is the average daily rate by hotel type?",
        "expect": "answer",
        "category": "Pricing analysis",
        "note": "ADR comparison between Resort Hotel and City Hotel",
    },
    {
        "question": "How does the cancellation rate compare between direct bookings and online travel agents?",
        "expect": "answer",
        "category": "Channel comparison",
        "note": "Direct vs Online TA - key Expedia insight",
    },
    {
        "question": "How has average daily rate changed across the years in the dataset?",
        "expect": "answer",
        "category": "Time-series - should trigger chart",
        "note": "ADR by year - should return chart_type=line",
    },
    # Guardrail tests — must be rejected
    {
        "question": "What is the current price of Expedia stock?",
        "expect": "reject",
        "category": "Guardrail - finance",
        "note": "Stock price - must be rejected",
    },
    {
        "question": "Tell me a joke about hotels",
        "expect": "reject",
        "category": "Guardrail - irrelevant",
        "note": "Off-topic request - must be rejected",
    },
    {
        "question": "What will hotel cancellation rates be in 2030?",
        "expect": "reject",
        "category": "Guardrail - prediction",
        "note": "Future prediction beyond the data - must be rejected",
    },
]

# Known valid tables
KNOWN_TABLES = ["BOOKINGS\".\"PUBLIC\".\"INFO", "\"INFO\""]


# Rule-based checks

def rule_checks(tc: dict, result: dict, elapsed: float) -> dict:
    checks = {}
    outcome = result.get("type", "error")

    checks["correct_outcome"] = (outcome == tc["expect"])
    checks["response_under_60s"] = (elapsed < 60)

    if outcome == "answer" and result.get("sql"):
        sql = result["sql"]
        checks["sql_uses_real_table"] = any(t in sql for t in KNOWN_TABLES)
    elif outcome == "answer":
        checks["sql_uses_real_table"] = False
    else:
        checks["sql_uses_real_table"] = None

    if outcome == "answer":
        checks["has_data"] = (len(result.get("rows", [])) > 0)
    else:
        checks["has_data"] = None

    if outcome == "answer":
        checks["has_meaningful_response"] = (len(result.get("message", "")) > 20)
    else:
        checks["has_meaningful_response"] = None

    if tc["expect"] == "reject":
        checks["no_sql_on_reject"] = (result.get("sql") is None)
    else:
        checks["no_sql_on_reject"] = None

    # Chart check: time-series questions should return a chart_type of line
    if "Time-series" in tc.get("category", "") and outcome == "answer":
        checks["chart_triggered"] = (result.get("chart_type") == "line")
    else:
        checks["chart_triggered"] = None

    return checks


# LLM-as-judge for answer quality (relevance, groundedness, clarity)

JUDGE_SYSTEM = """You are an impartial evaluator for a hotel booking analytics question-answering agent.
You will be given a user question and the agent's response.
Score the response on three criteria, each from 1 to 5, and a simple one-sentence reasoning.

Criteria:
- relevance:     Does the response directly address what the user asked? (1=off-topic, 5=perfect)
- groundedness:  Does it sound factually based on real booking data, not made up? (1=hallucinated, 5=well-grounded)
- clarity:       Is it clear, concise, and useful for a business analyst? (1=confusing, 5=very clear)

Respond ONLY with raw JSON, no markdown:
{"relevance": <1-5>, "groundedness": <1-5>, "clarity": <1-5>, "reasoning": "<one sentence>"}
"""


def judge_answer(question: str, answer: str, groq_client: Groq) -> dict:
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=200,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"Question: {question}\n\nAgent response: {answer}"},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"    [judge error] {e}")
        return None


# Calling the agent and measuring response time

def call_agent(base_url: str, question: str, client: httpx.Client) -> tuple[dict, float]:
    start = time.time()
    try:
        resp = client.post(
            f"{base_url}/chat",
            json={"message": question, "history": []},
            timeout=90,
        )
        elapsed = time.time() - start
        return resp.json(), elapsed
    except Exception as e:
        elapsed = time.time() - start
        return {"type": "error", "message": str(e)}, elapsed


# Building the evaluation report in md format

def build_report(results: list, base_url: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(results)

    correct_outcome = sum(1 for r in results if r["rules"].get("correct_outcome"))
    under_60 = sum(1 for r in results if r["rules"].get("response_under_60s"))
    avg_time = sum(r["elapsed"] for r in results) / total

    reject_results = [r for r in results if r["tc"]["expect"] == "reject"]
    correct_rejects = sum(1 for r in reject_results if r["result"]["type"] == "reject")

    chart_cases = [r for r in results if r["rules"].get("chart_triggered") is not None]
    correct_charts = sum(1 for r in chart_cases if r["rules"].get("chart_triggered"))

    judge_scores = [r["judge"] for r in results if r.get("judge")]
    avg_relevance     = sum(j["relevance"] for j in judge_scores) / len(judge_scores) if judge_scores else 0
    avg_groundedness  = sum(j["groundedness"] for j in judge_scores) / len(judge_scores) if judge_scores else 0
    avg_clarity       = sum(j["clarity"] for j in judge_scores) / len(judge_scores) if judge_scores else 0

    lines = [
        f"# Travel Insights Agent : Evaluation Report",
        f"",
        f"**Date:** {ts}  ",
        f"**Endpoint:** {base_url}  ",
        f"**Total test cases:** {total}  ",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Result |",
        f"|--------|--------|",
        f"| Correct outcome (answer vs reject) | {correct_outcome}/{total} ({100*correct_outcome//total}%) |",
        f"| Guardrail accuracy | {correct_rejects}/{len(reject_results)} off-topic questions correctly rejected |",
        f"| Chart triggered for time-series | {correct_charts}/{len(chart_cases)} |",
        f"| Responses under 60s | {under_60}/{total} |",
        f"| Average response time | {avg_time:.1f}s |",
        f"| LLM judge - avg relevance | {avg_relevance:.2f}/5 |",
        f"| LLM judge - avg groundedness | {avg_groundedness:.2f}/5 |",
        f"| LLM judge - avg clarity | {avg_clarity:.2f}/5 |",
        f"",
        f"---",
        f"",
        f"## Results by Test Case",
        f"",
    ]

    for i, r in enumerate(results, 1):
        tc = r["tc"]
        result = r["result"]
        rules = r["rules"]
        judge = r.get("judge")
        elapsed = r["elapsed"]

        outcome = result.get("type", "error")
        correct = "✅" if rules.get("correct_outcome") else "❌"

        lines.append(f"### {i}. {tc['question']}")
        lines.append(f"")
        lines.append(f"**Category:** {tc['category']}  ")
        lines.append(f"**Expected:** `{tc['expect']}` → **Got:** `{outcome}` {correct}  ")
        lines.append(f"**Response time:** {elapsed:.1f}s  ")
        if outcome == "answer" and result.get("chart_type"):
            lines.append(f"**Chart type returned:** `{result['chart_type']}`  ")
        lines.append(f"")

        if outcome == "answer":
            lines.append(f"**Agent answer:** {result.get('message','')[:300]}  ")
            lines.append(f"")
            if result.get("sql"):
                lines.append(f"```sql")
                lines.append(result["sql"][:400])
                lines.append(f"```")
                lines.append(f"")
        elif outcome == "reject":
            lines.append(f"**Rejection message:** {result.get('message','')[:200]}  ")
            lines.append(f"")
        elif outcome == "error":
            lines.append(f"**Error:** {result.get('message','')[:200]}  ")
            lines.append(f"")

        lines.append(f"**Rule checks:**")
        for check, val in rules.items():
            if val is None:
                continue
            icon = "✅" if val else "❌"
            lines.append(f"- {icon} {check.replace('_', ' ')}")
        lines.append(f"")

        if judge:
            lines.append(f"**LLM judge:** relevance {judge['relevance']}/5 · groundedness {judge['groundedness']}/5 · clarity {judge['clarity']}/5  ")
            lines.append(f"**Reasoning:** {judge.get('reasoning','')}  ")
            lines.append(f"")

        lines.append(f"---")
        lines.append(f"")

    failures = [r for r in results if not r["rules"].get("correct_outcome")]
    if failures:
        lines.append(f"## ⚠️ Failed Test Cases")
        lines.append(f"")
        for r in failures:
            lines.append(f"- **{r['tc']['question']}** — expected `{r['tc']['expect']}`, got `{r['result'].get('type')}`")
        lines.append(f"")

    return "\n".join(lines)


# Main call

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--out", default="eval_report.md")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    print(f"\n✈️  Travel Insights Agent Evaluation")
    print(f"   Endpoint : {base_url}")
    print(f"   Tests    : {len(TEST_CASES)}")
    print(f"   Judge    : {'disabled' if args.skip_judge else 'Groq Llama-3.3-70B'}")
    print(f"{'─'*55}\n")

    results = []
    with httpx.Client() as http:
        for i, tc in enumerate(TEST_CASES, 1):
            print(f"[{i:02d}/{len(TEST_CASES)}] {tc['question'][:60]}...")
            result, elapsed = call_agent(base_url, tc["question"], http)
            outcome = result.get("type", "error")
            print(f"       → {outcome} in {elapsed:.1f}s", end="")

            rules = rule_checks(tc, result, elapsed)
            correct = rules.get("correct_outcome")
            print(f" {'✅' if correct else '❌'}")

            judge = None
            if not args.skip_judge and outcome == "answer" and result.get("message"):
                print(f"       → judging...", end="", flush=True)
                judge = judge_answer(tc["question"], result["message"], groq_client)
                if judge:
                    print(f" relevance={judge['relevance']} groundedness={judge['groundedness']} clarity={judge['clarity']}")
                else:
                    print(f" [failed]")

            results.append({"tc": tc, "result": result, "rules": rules, "judge": judge, "elapsed": elapsed})
            time.sleep(0.5)

    report = build_report(results, base_url)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    total = len(results)
    correct = sum(1 for r in results if r["rules"].get("correct_outcome"))
    avg_time = sum(r["elapsed"] for r in results) / total

    print(f"\n{'─'*55}")
    print(f"✅ Correct outcomes : {correct}/{total}")
    print(f"⏱  Avg response time: {avg_time:.1f}s")
    print(f"📄 Report saved to  : {args.out}")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
