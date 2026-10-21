"""数据契约层：所有跨模块传递的对象都在这里定义，避免 prompt 与 API 各写一份。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ExamStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    CLOSED = "closed"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


QuestionType = Literal["single", "multiple", "judge"]


@dataclass
class Evidence:
    """一道题的溯源依据：从检索结果回填，不信任模型自己输出的页码。"""

    doc: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return {"doc": self.doc, "page": self.page, "text": self.text}


@dataclass
class Question:
    stem: str
    options: list[str]
    answer: list[str]          # 多选题也是列表，避免 R04 的「答案数量不定」
    qtype: QuestionType
    evidence: list[Evidence] = field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM
    knowledge_points: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stem": self.stem,
            "options": self.options,
            "answer": self.answer,
            "type": self.qtype,
            "evidence": [e.to_dict() for e in self.evidence],
            "difficulty": self.difficulty.value,
            "knowledge_points": self.knowledge_points,
        }

    def validate(self) -> list[str]:
        """生成后的规则校验：拦掉 LLM 最常见的几类错。"""
        errs: list[str] = []
        if not self.stem.strip():
            errs.append("题干为空")
        if self.qtype in ("single", "multiple") and len(self.options) < 2:
            errs.append("选项少于 2 个")
        if not self.answer:
            errs.append("答案为空")
        for a in self.answer:
            if self.options and a not in self.options:
                errs.append(f"答案 {a!r} 不在选项中")
        if self.qtype == "single" and len(self.answer) != 1:
            errs.append("单选题答案数量必须为 1")
        if not self.evidence:
            errs.append("缺少溯源依据")
        # R02：选项长度差异过大，等于变相暗示答案
        if len(self.options) >= 2:
            lens = [len(o) for o in self.options]
            if max(lens) > 2 * max(1, min(lens)):
                errs.append("选项长度差异过大，可能暗示答案")
        return errs


@dataclass
class Exam:
    id: str
    title: str
    collection: str
    questions: list[Question] = field(default_factory=list)
    status: ExamStatus = ExamStatus.DRAFT
    snapshot: list[dict] | None = None   # 发布时冻结，之后题库更新不影响已发布考试

    def to_dict(self, published: bool = False) -> dict:
        qs = self.snapshot if (published and self.snapshot) else [q.to_dict() for q in self.questions]
        return {
            "id": self.id,
            "title": self.title,
            "collection": self.collection,
            "status": self.status.value,
            "question_count": len(qs),
            "questions": qs,
        }
