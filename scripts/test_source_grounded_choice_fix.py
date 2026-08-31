from __future__ import annotations

from types import SimpleNamespace

from bankreg_trustrag.query_plan import PlannerOutput
from bankreg_trustrag.query_planner import _source_grounded_choice_plan
from bankreg_trustrag.retrieval_tools import _text_result


QUESTION = (
    "根据《寿险合同负债评估折现率曲线》，下列哪项表述正确？"
    "A.终极利率适用于40年以后的区间。"
    "B.列入名单的保险集团应当按照《保险公司偿付能力监管规则第19号：保险集团》有关规定编报保险集团偿付能力报告。"
    "C.其他符合保险集团定义的保险集团暂不编报偿付能力报告。"
    "D.中国人民保险集团股份有限公司属于应当编制保险集团偿付能力报告的保险控股型集团。"
)


class DummyHit:
    def __init__(self, evidence_id: str, content: str, rerank_score: float):
        self.evidence_id = evidence_id
        self.item = {"doc_id": "DOC_TEST", "content": content}
        self.rerank_score = rerank_score
        self.dense_score = 0.0
        self.fused_score = 0.0


def main() -> None:
    # 模拟“旧 Planner 错误拆成四个选项任务”的输出；修复后的后处理应强制
    # 把单一明确来源的监管选择题归一化为一个证据任务。
    compact = PlannerOutput.model_validate({
        "user_goal": "判断哪个选项符合指定附件原文",
        "answer_requirements": [{
            "id": "ar1",
            "question": QUESTION,
            "required_outputs": ["r1", "r2", "r3", "r4"],
        }],
        "retrieval_tasks": [
            {"id": f"r{i}", "query": QUESTION, "expected_information": f"核对选项{i}"}
            for i in range(1, 5)
        ],
        "operations": [],
        "requires_clarification": False,
    })
    plan = _source_grounded_choice_plan(QUESTION, compact)
    assert plan is not None
    assert len(plan.retrieval_tasks) == 1
    assert plan.answer_requirements[0].required_outputs == ["r1"]
    assert plan.retrieval_tasks[0].source_scope.document_title == "寿险合同负债评估折现率曲线"

    hits = [
        DummyHit(
            "p17",
            "保险公司按照本附件规定的即期基础利率曲线附加综合溢价，得到即期折现率曲线。"
            "保险合同负债评估中所使用的远期折现率曲线，由即期折现率曲线换算得到。",
            0.64,
        ),
        DummyHit(
            "p1",
            "附件1寿险合同负债评估折现率曲线根据《保险公司偿付能力监管规则第3号：寿险合同负债评估》"
            "第十九条规定，计算现金流现值所采用的折现率曲线由基础利率曲线加综合溢价形成。",
            0.85,
        ),
        DummyHit(
            "p3",
            "750日移动平均国债收益率曲线 0 < t ≤ 20；终极利率过渡曲线 20 < t ≤ 40；"
            "终极利率 t > 40。其中，t表示年度；终极利率暂定为4.5%。",
            0.57,
        ),
        DummyHit("p9", "r*t为在t年度750天移动平均国债收益率曲线的数值。", 0.40),
    ]
    index = SimpleNamespace(doc_by_id={
        "DOC_TEST": {
            "title": "中国银保监会关于实施_保险公司偿付能力监管规则(II)_有关事项的通知_附件1:寿险合同负债评估的折现率曲线",
            "document_type": "pdf",
        }
    })
    execution = _text_result(plan.retrieval_tasks[0], hits, index)
    assert execution.result.status == "resolved"
    assert execution.result.selected is not None
    assert execution.result.selected.evidence_ids == ["p3"], execution.result.model_dump()

    print("PASS: 单一来源选择题只生成1个检索任务")
    print("PASS: 正确证据 p3 被选中，而不是 candidates[0]/p17")
    print("selected score =", execution.result.selected.score)


if __name__ == "__main__":
    main()
