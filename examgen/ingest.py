"""知识入库：把制度文档切成「条款级」chunk 并建索引。

设计要点（对应优化记录 R07）：
    制度类文档按固定长度切块会把一条款拦腰截断，导致检索到的片段语义不完整。
    这里优先按「条款标题」切分（第X条 / X.X / 一、二、三 …），只有超长条款才二次按句子切。
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from rank_bm25 import BM25Okapi

INDEX_DIR = Path(".cache/index")

# 条款起始的常见模式：第X条 / 1.2 / 1.2.3 / （一）/ 一、
_CLAUSE_RE = re.compile(
    r"^\s*(?:"
    r"第[一二三四五六七八九十百零\d]+条"
    r"|[（(][一二三四五六七八九十]+[)）]"
    r"|[一二三四五六七八九十]+、"
    r"|\d+(?:\.\d+){0,3}[、.\s]"
    r")"
)

_SENT_RE = re.compile(r"(?<=[。！？；\n])")


@dataclass
class Chunk:
    doc: str
    page: int
    text: str
    clause: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def load_text(path: Path) -> list[tuple[int, str]]:
    """按页读出 [(page_no, text)]。PDF 走 pypdf，纯文本/markdown 直接读。"""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    return [(1, path.read_text(encoding="utf-8"))]


def split_by_clause(pages: list[tuple[int, str]], doc: str = "", max_len: int = 700) -> list[Chunk]:
    """先按条款切，超长条款再按句子二次切。"""
    chunks: list[Chunk] = []
    for page_no, text in pages:
        buf: list[str] = []
        clause = ""

        def flush() -> None:
            body = "".join(buf).strip()
            if body:
                chunks.append(Chunk(doc=doc, page=page_no, text=body, clause=clause))
            buf.clear()

        for line in text.splitlines():
            line_s = line.strip()
            if not line_s:
                buf.append("\n")
                continue
            if _CLAUSE_RE.match(line_s) and buf:
                flush()
                clause = line_s[:30]
            elif not clause:
                clause = line_s[:30]
            buf.append(line_s + "\n")
        flush()

    # 超长条款按句子二次切，保留 overlap 以免切断因果
    refined: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= max_len:
            refined.append(c)
            continue
        parts = [p for p in _SENT_RE.split(c.text) if p]
        cur = ""
        for p in parts:
            if len(cur) + len(p) > max_len and cur:
                refined.append(Chunk(c.doc, c.page, cur.strip(), c.clause))
                cur = cur[-80:]  # overlap
            cur += p
        if cur.strip():
            refined.append(Chunk(c.doc, c.page, cur.strip(), c.clause))
    return refined


def tokenize(text: str) -> list[str]:
    """中文按字 + 英文数字按词，够用且无额外依赖。"""
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())


def build(collection: str, chunks: list[Chunk]) -> Path:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    corpus = [tokenize(c.text) for c in chunks]
    payload = {
        "collection": collection,
        "chunks": [c.to_dict() for c in chunks],
        "bm25": BM25Okapi(corpus),
    }
    out = INDEX_DIR / f"{collection}.pkl"
    out.write_bytes(pickle.dumps(payload))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="导入制度文档并建立检索索引")
    ap.add_argument("--input", required=True, help="PDF / md / txt 路径")
    ap.add_argument("--collection", required=True, help="知识库名，如 hr-2026")
    ap.add_argument("--max-len", type=int, default=700)
    args = ap.parse_args()

    path = Path(args.input)
    chunks = split_by_clause(load_text(path), doc=path.name, max_len=args.max_len)
    out = build(args.collection, chunks)
    print(json.dumps({"collection": args.collection, "chunks": len(chunks), "index": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
