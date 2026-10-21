# AI Exam Generator

基于企业内部制度文档自动生成试题：**知识入库 → 试题生成 → 审核发布 → 在线考试 → 评分统计**。

核心思路不是「让 AI 写题」，而是**把专家从出题里解放出来，只保留「审核」这一个质量节点**。

## 业务命题

| 传统方式 | 问题 |
|---|---|
| 专家人工出题 | 对出题人要求高，且占用专家精力 |
| 题库更新慢 | 学员背旧题，达不到「督促关注最新制度」的目的 |

**改造后**：AI 负责出题（可批量、可随制度更新），专家/培训负责人只做**审核发布**。

## 系统架构（4 模块）

```text
┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────────┐
│ 知识入库  │──▶│  试题生成     │──▶│ 审核发布  │──▶│ 在线考试/评分 │
│ 切块+向量 │   │ 检索→生成→依据 │   │ 快照发布  │   │  交卷→统计   │
└──────────┘   └──────────────┘   └──────────┘   └──────────────┘
```

**发布快照机制**（关键设计）：发布时把当时的题目与依据**冻结成快照**。后续题库更新不会影响已发布的考试，保证同一次考试对所有考生口径一致。

## 出题链路

```text
制度文档 ─▶ 切块 ─▶ 检索 Top-12 ─▶ 精筛保留 Top-6 ─▶ LLM 生成 + 强制附依据
```

**为什么是 12 → 6**：检索 12 段保证召回，但全塞进 prompt 会稀释注意力、拉高成本。实测保留 6 段时**题目与制度的相关性和成本平衡最好**——多出来的段落大多是同义重复。

每道题强制携带 `evidence`（制度出处 + 原文片段），审核者可直接核对，也把幻觉挡在审核环节之前。

```python
@dataclass
class Question:
    stem: str
    options: list[str]
    answer: str
    evidence: list[Evidence]     # 强制溯源
    difficulty: str
    knowledge_points: list[str]
```

## API 与错误处理

| 端点 | 方法 | 说明 |
|---|---|---|
| `/exams` | POST | 创建考试 |
| `/exams/{id}/publish` | POST | 发布（生成快照） |
| `/exams/{id}/submit` | POST | 交卷评分 |
| `/exams/{id}/stats` | GET | 成绩统计 |

错误码是刻意区分的：

- **409 Conflict** —— 重复发布 / 重复交卷。客户端应视为「已经做过」，提示用户刷新而不是重试
- **422 Unprocessable** —— 输入非法（如题目数量超限、难度分布不合法）。需要在**生成前**校验，而不是等 LLM 产出错题再报错

```python
if exam.status == ExamStatus.PUBLISHED:
    raise HTTPException(409, "考试已发布，不能重复发布")
if not (1 <= payload.question_count <= 100):
    raise HTTPException(422, "题目数量需在 1-100 之间")
```

## 效果优化记录（R01–R08）

三批开发过程中沉淀的典型问题与修法：

| 编号 | 问题 | 修法 |
|---|---|---|
| R01 | 题目过于泛化，脱离制度原文 | prompt 增加「必须引用给定资料，禁止常识发挥」约束 |
| R02 | 选项长度不一致，暗示答案 | 约束各选项字数接近 |
| R03 | 同一知识点重复出题 | 生成后按 `knowledge_points` 去重 |
| R04 | 多选题答案数量不定 | 明确在 schema 里约束 `answer` 为列表 |
| R05 | 难度分布失衡 | 生成时按难度配额抽样，而非事后筛 |
| R06 | 依据页码错误 | 依据改为从检索结果回填，不信任模型输出的页码 |
| R07 | 长文档切块切断条款 | 按条款标题切分，而非固定长度 |
| R08 | 成本偏高 | 12→6 精筛 + 批量生成复用上下文 |

## 评估

客观指标走规则校验（答案是否在选项中、依据是否命中原文），主观指标（题目表述是否自然、干扰项是否合理）用 LLM-as-Judge 抽样。

```bash
python -m examgen.evaluate --cases cases.jsonl
```

## 快速开始

```bash
pip install -r requirements.txt
export DASHSCOPE_API_KEY=sk-xxx

# 1) 导入制度文档
python -m examgen.ingest --input ./docs/policy.pdf --collection hr-2026

# 2) 生成试题
python -m examgen.generate --collection hr-2026 --count 20 --difficulty mixed

# 3) 启动服务
uvicorn examgen.api:app --reload
```

## 参考

- 方法论参考 DeepSeek Harness 协作流程（需求讨论 → 场景 → 选型 → 架构 → 方案审阅 → 分批开发 → 评估优化 → 整合验收），本仓库实现的是其中的出题与发布链路，并补充了 API 错误语义与发布快照设计。

## License

MIT
