"""试题生成：检索 Top-12 → 精筛 Top-6 → LLM 结构化生成 → 规则校验 → 去重。"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .ingest import INDEX_DIR, Chunk, tokenize
from .schema import Difficulty, Evidence, Question

RETRIEVE_K = 12
KEEP_K = 6

GEN_PROMPT = """你是企业培训出题专家。只能依据下面提供的【制度原文】出题，禁止使用常识或外部知识发挥。

【制度原文】
{context}

【要求】
1. 每题必须能从原文中找到直接依据，并在 evidence 中给出 doc / page / 原文片段。
2. 四个选项（判断题为「正确/错误」），各选项字数尽量接近，不要出现「以上都对」这类暗示项。
3. 单选题 answer 长度为 1，多选题 answer 为列表且至少 2 个。
4. 给出 difficulty（easy/medium/hard）与 knowledge_points（1-3 个短语）。
5. 难度按配额：{quota}

evidence 用检索片段的编号引用，格式为 [{{"ref": 1, "quote": "原文片段"}}]，不要自己编页码。

输出严格为 JSON 数组，每个元素字段：stem, options, answer, type, difficulty, knowledge_points, evidence。
"""

_QUOTA = {
    "easy": {"easy": 1.0, "medium": 0.0, "hard": 0.0},
    "medium": {"easy": 0.2, "medium": 0.6, "hard": 0.2},
    "hard": {"easy": 0.0, "medium": 0.3, "hard": 0.7},
    "mixed": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
}


def load_index(collection: str) -> tuple[list[Chunk], BM25Okapi]:
    p = INDEX_DIR / f"{collection}.pkl"
    if not p.exists():
        raise SystemExit(f"索引不存在，先跑：python -m examgen.ingest --input <文件> --collection {collection}")
    data = pickle.loads(p.read_bytes())
    return [Chunk(**c) for c in data["chunks"]], data["bm25"]


def retrieve(query: str, chunks: list[Chunk], bm25: BM25Okapi, k: int = RETRIEVE_K) -> list[Chunk]:
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in order]


def keep_top(chunks: list[Chunk], k: int = KEEP_K) -> list[Chunk]:
    """12 → 6：去掉同义重复段（简易去重：字符 3-gram 交集过高的丢弃）。"""
    kept: list[Chunk] = []
    for c in chunks:
        sig = _sig(c.text)
        if any(_jaccard(sig, _sig(k2.text)) > 0.6 for k2 in kept):
            continue
        kept.append(c)
        if len(kept) >= k:
            break
    return kept or chunks[:k]


def _sig(text: str) -> set[str]:
    return {text[i : i + 3] for i in range(max(1, len(text) - 2))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i + 1}] doc={c.doc} page={c.page}\n{c.text}" for i, c in enumerate(chunks))


def _quota_text(difficulty: str, count: int) -> str:
    q = _QUOTA.get(difficulty, _QUOTA["mixed"])
    return "、".join(f"{k} 约 {int(v * count)} 题" for k, v in q.items() if v)


DEMO_MODE = False


def call_llm(prompt: str, model: str = "qwen-plus") -> str:
    """DashScope OpenAI 兼容接口。

    没装 openai 或没配 key 时进入 DEMO 模式（返回空数组），链路照样跑通，方便离线验证检索与校验逻辑。
    """
    global DEMO_MODE
    try:
        from openai import OpenAI
    except ImportError:
        DEMO_MODE = True
        return "[]"
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        DEMO_MODE = True
        return "[]"

    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return resp.choices[0].message.content or "[]"


def _coerce_evidence(raw_ev: list, chunks: list[Chunk], page_map: dict[int, Chunk]) -> list[Evidence]:
    """ evidence 一律回填检索结果，不信任模型给的页码（R06）。

    模型可能给 dict（ref/quote/doc/page）、纯数字字符串、或一段原文，三种都要能吃下。
    """
    if not chunks:
        return []
    ev: list[Evidence] = []
    for e in raw_ev:
        src: Chunk | None = None
        quote = ""
        if isinstance(e, dict):
            idx = e.get("ref")
            src = page_map.get(int(idx)) if str(idx).isdigit() else None
            quote = e.get("quote") or e.get("text") or ""
            if src is None and e.get("text"):
                src = _match_chunk(e["text"], chunks)
        elif isinstance(e, (int, str)):
            s = str(e).strip()
            if s.isdigit():
                src = page_map.get(int(s))
            else:
                src = _match_chunk(s, chunks) or chunks[0]
                quote = s
        if src is None:
            continue
        ev.append(Evidence(doc=src.doc, page=src.page, text=(quote or src.text)[:200]))
    return ev


_TYPE_ALIAS = {
    "single": "single", "single_choice": "single", "单选": "single", "单选题": "single", "radio": "single",
    "multiple": "multiple", "multi": "multiple", "multiple_choice": "multiple", "多选": "multiple",
    "多选题": "multiple", "checkbox": "multiple",
    "judge": "judge", "判断题": "judge", "true_false": "judge", "boolean": "judge",
}


def _coerce_type(value: str) -> str:
    return _TYPE_ALIAS.get(str(value).strip().lower(), "single")


def _coerce_answer(ans, options: list[str], qtype: str) -> list[str]:
    """模型给出的答案形态很不稳定：选项文本 / 字母 b / ['b','d'] / 布尔值，都要归一到选项文本。"""
    items = [ans] if isinstance(ans, str) else list(ans or [])
    out: list[str] = []
    for a in items:
        s = str(a).strip()
        if s in options:
            out.append(s)
            continue
        m = re.match(r"^([A-Da-d])[.、)）:：]?$", s)          # 字母选项
        if m:
            i = ord(m.group(1).lower()) - ord("a")
            if 0 <= i < len(options):
                out.append(options[i])
                continue
        if qtype == "judge" or s.lower() in ("true", "false", "yes", "no", "正确", "错误", "对", "错"):
            out.append("正确" if s.lower() in ("true", "yes", "正确", "对", "1") else "错误")
            continue
        for o in options:                                     # 模糊兜底
            if s and (s in o or o in s):
                out.append(o)
                break
    return out


def _match_chunk(text: str, chunks: list[Chunk]) -> Chunk | None:
    head = text.strip()[:20]
    if not head:
        return None
    for c in chunks:
        if head in c.text:
            return c
    return None


def parse_questions(raw: str, chunks: list[Chunk]) -> list[Question]:
    """解析模型输出；evidence 一律从检索结果回填，不信任模型给的页码（R06）。"""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("questions", [])
    page_map = {i + 1: c for i, c in enumerate(chunks)}

    out: list[Question] = []
    for item in data:
        ev = _coerce_evidence(item.get("evidence", []), chunks, page_map)
        if not ev:
            continue
        options = item.get("options", []) or []
        qtype = _coerce_type(item.get("type", "single"))
        try:
            difficulty = Difficulty(item.get("difficulty", "medium"))
        except ValueError:
            difficulty = Difficulty.MEDIUM
        out.append(
            Question(
                stem=item.get("stem", "").strip(),
                options=options,
                answer=_coerce_answer(item.get("answer", []), options, qtype),
                qtype=qtype,
                evidence=ev,
                difficulty=difficulty,
                knowledge_points=item.get("knowledge_points", []) or [],
            )
        )
    return out


def dedupe(questions: list[Question]) -> list[Question]:
    """R03：同一知识点重复出题。"""
    seen: set[str] = set()
    out: list[Question] = []
    for q in questions:
        key = "|".join(sorted(q.knowledge_points)) or q.stem[:40]
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def generate(
    collection: str,
    topic: str = "制度要点",
    count: int = 20,
    difficulty: str = "mixed",
    model: str = "qwen-plus",
) -> list[Question]:
    chunks, bm25 = load_index(collection)
    hits = retrieve(topic, chunks, bm25, k=RETRIEVE_K)
    kept = keep_top(hits, k=KEEP_K)

    raw = call_llm(GEN_PROMPT.format(context=_context(kept), quota=_quota_text(difficulty, count)), model=model)
    qs = parse_questions(raw, kept)
    qs = [q for q in qs if not q.validate()]
    qs = dedupe(qs)
    return qs[:count]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--topic", default="制度要点")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--difficulty", default="mixed", choices=list(_QUOTA))
    ap.add_argument("--model", default="qwen-plus")
    ap.add_argument("--out", default="questions.json")
    args = ap.parse_args()

    if not (1 <= args.count <= 100):
        raise SystemExit("题目数量需在 1-100 之间")  # 生成前校验，别等 LLM 出完再报 422

    # R05：按难度配额抽样，而不是生成完再筛
    qs = generate(args.collection, args.topic, args.count, args.difficulty, args.model)
    Path(args.out).write_text(json.dumps([q.to_dict() for q in qs], ensure_ascii=False, indent=2), encoding="utf-8")
    if DEMO_MODE:
        print("DEMO 模式：未配置 DASHSCOPE_API_KEY / 未安装 openai，仅验证检索与校验链路")
    print(f"生成 {len(qs)} 题 -> {args.out}")
    for q in qs:
        flag = "OK " if not q.validate() else "BAD"
        print(f"  [{flag}] {q.difficulty.value:<6} {q.stem[:36]}...")


if __name__ == "__main__":
    main()
