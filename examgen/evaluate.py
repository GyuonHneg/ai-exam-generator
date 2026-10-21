"""离线评估：客观指标走规则校验，主观指标交给 LLM-as-Judge 抽样。

    python -m examgen.evaluate --cases cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def objective_metrics(questions: list[dict], index_chunks: list[str]) -> dict:
    """不需要 LLM 的硬指标：答案合法率、依据命中率、去重率。"""
    n = len(questions) or 1
    answer_ok = sum(
        1
        for q in questions
        if q.get("answer")
        and (not q.get("options") or all(a in q["options"] for a in q["answer"]))
    )
    evidence_hit = 0
    for q in questions:
        for e in q.get("evidence", []):
            if any(e.get("text", "")[:50] in c for c in index_chunks):
                evidence_hit += 1
                break
    dedup_rate = len({q["stem"] for q in questions}) / n
    return {
        "total": len(questions),
        "answer_valid_rate": round(answer_ok / n, 3),
        "evidence_hit_rate": round(evidence_hit / n, 3),
        "unique_rate": round(dedup_rate, 3),
    }


def judge(questions: list[dict], sample: int = 5) -> list[dict]:
    """主观指标：表述自然度 + 干扰项合理性，抽样打分 1-5。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return [{"note": "未配置 DASHSCOPE_API_KEY，跳过 LLM-as-Judge"}]
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    out = []
    for q in questions[:sample]:
        prompt = (
            "你是企业培训审题专家。给下面这道题的『表述自然度』和『干扰项合理性』各打 1-5 分，"
            "并给一句修改建议。输出 JSON：{natural:int, distractor:int, advice:str}\n\n"
            + json.dumps(q, ensure_ascii=False)
        )
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        try:
            out.append(json.loads(resp.choices[0].message.content))
        except json.JSONDecodeError:
            out.append({"stem": q.get("stem", "")[:30], "error": "judge 输出无法解析"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="cases.jsonl", help="每行一个 {collection, topic, count}")
    ap.add_argument("--sample", type=int, default=5)
    args = ap.parse_args()

    from .generate import generate, load_index

    for case in load_cases(Path(args.cases)):
        qs = [q.to_dict() for q in generate(**case)]
        try:
            chunks, _bm25 = load_index(case["collection"])
            corpus = [c.text for c in chunks]
        except SystemExit:
            corpus = []
        metrics = objective_metrics(qs, corpus)
        print(json.dumps({"case": case, **metrics}, ensure_ascii=False))
        if args.sample:
            print("  judge:", json.dumps(judge(qs, args.sample), ensure_ascii=False))


if __name__ == "__main__":
    main()
