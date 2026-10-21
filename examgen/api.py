"""对外服务：创建 / 发布（快照）/ 交卷评分 / 统计。

错误语义是刻意区分的：
    409 = 状态冲突（重复发布、重复交卷）——客户端应视为「已做过」，提示刷新而非重试
    422 = 输入非法（题目数超限、难度分布不合法）——必须在生成前校验，而不是等 LLM 产出错题再报错
"""

from __future__ import annotations

import uuid
from collections import Counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generate import generate
from .schema import Exam, ExamStatus, Question

app = FastAPI(title="AI Exam Generator", version="0.1.0")

_EXAMS: dict[str, Exam] = {}
_SUBMISSIONS: dict[str, dict[str, list[str]]] = {}   # exam_id -> {user: [answer..]}


class CreateExamIn(BaseModel):
    title: str
    collection: str
    topic: str = "制度要点"
    question_count: int = Field(default=20, ge=1, le=100)
    difficulty: str = "mixed"


class SubmitIn(BaseModel):
    user: str
    answers: dict[str, list[str]] = Field(default_factory=dict)   # question_index -> [option..]


def _get(exam_id: str) -> Exam:
    exam = _EXAMS.get(exam_id)
    if not exam:
        raise HTTPException(404, "考试不存在")
    return exam


@app.post("/exams")
def create_exam(payload: CreateExamIn) -> dict:
    """创建考试（草稿态）。生成是同步的，题量与难度在这里先做边界校验。"""
    if payload.difficulty not in ("easy", "medium", "hard", "mixed"):
        raise HTTPException(422, "difficulty 必须是 easy/medium/hard/mixed")
    qs = generate(
        collection=payload.collection,
        topic=payload.topic,
        count=payload.question_count,
        difficulty=payload.difficulty,
    )
    exam = Exam(id=uuid.uuid4().hex[:8], title=payload.title, collection=payload.collection, questions=qs)
    _EXAMS[exam.id] = exam
    return {"id": exam.id, "status": exam.status.value, "question_count": len(qs)}


@app.get("/exams/{exam_id}")
def get_exam(exam_id: str) -> dict:
    exam = _get(exam_id)
    return exam.to_dict(published=True)


@app.post("/exams/{exam_id}/publish")
def publish(exam_id: str) -> dict:
    """发布：把当时的题目与依据冻结为快照，之后题库更新不影响已发布考试。"""
    exam = _get(exam_id)
    if exam.status is ExamStatus.PUBLISHED:
        raise HTTPException(409, "考试已发布，不能重复发布")     # 状态冲突 → 409
    if not exam.questions:
        raise HTTPException(422, "没有题目，无法发布")            # 输入/数据不合法 → 422
    exam.snapshot = [q.to_dict() for q in exam.questions]
    exam.status = ExamStatus.PUBLISHED
    return {"id": exam.id, "status": exam.status.value, "snapshot_size": len(exam.snapshot)}


@app.post("/exams/{exam_id}/submit")
def submit(exam_id: str, payload: SubmitIn) -> dict:
    exam = _get(exam_id)
    if exam.status is not ExamStatus.PUBLISHED:
        raise HTTPException(422, "考试未发布，不能交卷")
    users = _SUBMISSIONS.setdefault(exam_id, {})
    if payload.user in users:
        raise HTTPException(409, "该用户已交卷")                  # 重复提交 → 409
    if not payload.answers:
        raise HTTPException(422, "答卷为空")

    source = exam.snapshot or [q.to_dict() for q in exam.questions]
    correct = 0
    detail = []
    for idx, q in enumerate(source):
        picked = set(payload.answers.get(str(idx), []))
        right = set(q["answer"])
        ok = picked == right
        correct += int(ok)
        detail.append({"index": idx, "correct": ok, "expected": sorted(right)})
    users[payload.user] = [str(i) for i in range(len(source))]
    return {"user": payload.user, "correct": correct, "total": len(source), "detail": detail}


@app.get("/exams/{exam_id}/stats")
def stats(exam_id: str) -> dict:
    exam = _get(exam_id)
    source = exam.snapshot or [q.to_dict() for q in exam.questions]
    users = _SUBMISSIONS.get(exam_id, {})
    return {
        "id": exam.id,
        "status": exam.status.value,
        "question_count": len(source),
        "submitted": len(users),
        "difficulty_dist": dict(Counter(q["difficulty"] for q in source)),
    }


def _seed_demo() -> None:
    """无 LLM key 时的演示数据，保证服务能起来、接口可联调。"""
    demo = [
        Question(
            stem="员工年假申请应至少提前多少个工作日提交？",
            options=["1 个工作日", "3 个工作日", "5 个工作日", "10 个工作日"],
            answer=["3 个工作日"],
            qtype="single",
            difficulty="easy",
            knowledge_points=["年假申请"],
        )
    ]
    exam = Exam(id="demo0001", title="演示考试", collection="demo", questions=demo, status=ExamStatus.PUBLISHED)
    exam.snapshot = [q.to_dict() for q in demo]
    _EXAMS[exam.id] = exam


_seed_demo()
