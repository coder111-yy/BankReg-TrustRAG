import json
import re
from types import SimpleNamespace

from bankreg_trustrag.llm_client import LLMClient, LLMClientConfig
from bankreg_trustrag.query_plan import PlannerOutput, QueryPlan
from bankreg_trustrag.query_planner import QueryPlanner


class _Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


def _plan_payload(question="查询总资产"):
    return {
        "original_query": question,
        "user_goal": "查询总资产",
        "answer_requirements": [{"id": "ar1", "question": "总资产是多少", "required_outputs": ["r1"]}],
        "entities": {"indicators": ["总资产"], "periods": ["2024-09"]},
        "retrieval_tasks": [{
            "id": "r1",
            "query": "2024年9月保险业总资产",
            "expected_information": "总资产数值",
            "source_scope": {"year": 2024, "month": 9},
            "semantic_constraints": {"indicator": "总资产", "period": "2024-09"},
            "expected_value_type": "number",
            "expected_unit": "亿元",
        }],
        "operations": [],
        "requires_multiple_sources": False,
        "requires_table_retrieval": True,
        "requires_calculation": False,
        "requires_clarification": False,
    }


def _compact_plan_payload():
    return {
        "user_goal": "查询总资产",
        "answer_requirements": [
            {"id": "ar1", "question": "总资产是多少", "required_outputs": ["r1"]}
        ],
        "retrieval_tasks": [{
            "id": "r1",
            "query": "2024年9月保险业总资产",
            "expected_information": "总资产数值",
            "indicator": "总资产",
            "period": "2024-09",
            "expected_value_type": "number",
            "expected_unit": "亿元",
        }],
        "operations": [],
        "requires_clarification": False,
    }


def _cross_file_compact_payload():
    return {
        "user_goal": "计算两类公司保费合计及与全国总数的差额",
        "answer_requirements": [
            {"id": "ar1", "question": "合计是多少", "required_outputs": ["calc1"]},
            {"id": "ar2", "question": "相差多少", "required_outputs": ["calc2"]},
        ],
        "retrieval_tasks": [
            {"id": "r1", "query": "人身险保费", "expected_information": "人身险", "indicator": "原保险保费收入", "period": "2023-10", "expected_value_type": "number"},
            {"id": "r2", "query": "财产险保费", "expected_information": "财产险", "indicator": "原保险保费收入", "period": "2023-10", "expected_value_type": "number"},
            {"id": "r3", "query": "全国总数", "expected_information": "全国", "indicator": "原保险保费收入", "period": "2023-10", "expected_value_type": "number"},
        ],
        "operations": [
            {"type": "sum", "output_id": "calc1", "inputs": ["r1", "r2"]},
            {"type": "subtract", "output_id": "calc2", "inputs": ["r3", "calc1"], "absolute": True},
        ],
        "requires_clarification": False,
    }


def _explicit_source_cross_file_payload_with_guessed_unit():
    return {
        "user_goal": "计算两类公司保费合计及与全国合计的差额",
        "answer_requirements": [
            {"id": "ar1", "question": "两类公司合计是多少", "required_outputs": ["calc1"]},
            {"id": "ar2", "question": "与全国合计相差多少", "required_outputs": ["calc2"]},
        ],
        "retrieval_tasks": [
            {
                "id": "r1", "query": "2023年10月人身险公司原保险保费收入",
                "expected_information": "人身险公司原保险保费收入", "indicator": "原保险保费收入",
                "institution": "人身险公司", "period": "2023年10月",
                "source_hint": "《2023年10月人身险公司经营情况表》",
                "expected_value_type": "number", "expected_unit": "万元",
            },
            {
                "id": "r2", "query": "2023年10月财产保险公司原保险保费收入",
                "expected_information": "财产保险公司原保险保费收入", "indicator": "原保险保费收入",
                "institution": "财产保险公司", "period": "2023年10月",
                "source_hint": "《2023年10月财产保险公司经营情况表》",
                "expected_value_type": "number", "expected_unit": "万元",
            },
            {
                "id": "r3", "query": "全国各地区原保险保费收入表中的全国合计",
                "expected_information": "全国原保险保费收入合计", "indicator": "原保险保费收入",
                "period": "2023年10月", "row_label": "全国合计",
                "column_label": "原保险保费收入",
                "source_hint": "《2023年10月全国各地区原保险保费收入情况表》",
                "expected_value_type": "number", "expected_unit": "万元",
            },
        ],
        "operations": [
            {"type": "sum", "output_id": "calc1", "inputs": ["r1", "r2"]},
            {"type": "subtract", "output_id": "calc2", "inputs": ["calc1", "r3"], "absolute": True},
        ],
        "requires_clarification": False,
    }


def test_llm_client_retries_invalid_structured_output(monkeypatch):
    responses = iter([_Response("not-json"), _Response(json.dumps(_plan_payload(), ensure_ascii=False))])
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs["json"])
        return next(responses)

    monkeypatch.setattr("bankreg_trustrag.llm_client.httpx.post", fake_post)
    client = LLMClient(LLMClientConfig(provider="openai_compatible", model="qwen", base_url="http://local/v1", max_retries=1))

    result = client.structured([{"role": "user", "content": "plan"}], QueryPlan, temperature=0.1)

    assert result.status == "ok"
    assert result.attempts == 2
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["response_format"]["type"] == "json_object"
    assert "必须严格符合的JSON Schema" in calls[1]["messages"][-1]["content"]
    assert '"required_outputs"' in calls[1]["messages"][-1]["content"]


def test_llm_client_accepts_wrapped_python_dict_from_provider(monkeypatch):
    malformed_but_unambiguous = "模型说明：\n```json\n" + repr(_plan_payload()) + "\n```"
    monkeypatch.setattr(
        "bankreg_trustrag.llm_client.httpx.post",
        lambda *args, **kwargs: _Response(malformed_but_unambiguous),
    )
    client = LLMClient(LLMClientConfig(
        provider="openai_compatible",
        model="deepseek",
        base_url="http://local/v1",
    ))

    result = client.structured(
        [{"role": "user", "content": "plan"}],
        QueryPlan,
        temperature=0.0,
    )

    assert result.status == "ok"
    assert result.value is not None
    assert result.value.user_goal == "查询总资产"
    assert result.attempts == 1


def test_deepseek_structured_request_disables_thinking(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs["json"])
        return _Response(json.dumps(_compact_plan_payload(), ensure_ascii=False))

    monkeypatch.setattr("bankreg_trustrag.llm_client.httpx.post", fake_post)
    client = LLMClient(LLMClientConfig(
        provider="deepseek",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    ))

    result = client.structured(
        [{"role": "user", "content": "plan"}],
        PlannerOutput,
        temperature=0.0,
        prefer_json_schema=False,
    )

    assert result.status == "ok"
    assert calls[0]["thinking"] == {"type": "disabled"}


def test_llm_client_accepts_compatibility_completion_text(monkeypatch):
    class CompletionResponse(_Response):
        def json(self):
            return {"choices": [{"text": json.dumps(_plan_payload(), ensure_ascii=False)}]}

    monkeypatch.setattr(
        "bankreg_trustrag.llm_client.httpx.post",
        lambda *args, **kwargs: CompletionResponse("unused"),
    )
    client = LLMClient(LLMClientConfig(
        provider="openai_compatible",
        model="local",
        base_url="http://local/v1",
    ))

    result = client.structured(
        [{"role": "user", "content": "plan"}],
        QueryPlan,
        temperature=0.0,
    )

    assert result.status == "ok"


def test_llm_client_repairs_unquoted_object_keys(monkeypatch):
    payload = json.dumps(_plan_payload(), ensure_ascii=False)
    malformed = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)":', r'\1:', payload)
    monkeypatch.setattr(
        "bankreg_trustrag.llm_client.httpx.post",
        lambda *args, **kwargs: _Response(malformed),
    )
    client = LLMClient(LLMClientConfig(
        provider="openai_compatible",
        model="deepseek",
        base_url="http://local/v1",
    ))

    result = client.structured(
        [{"role": "user", "content": "plan"}],
        QueryPlan,
        temperature=0.0,
    )

    assert result.status == "ok"
    assert result.value is not None
    assert result.value.user_goal == "查询总资产"


def test_query_planner_preserves_application_original_question(monkeypatch):
    payload = _compact_plan_payload()
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs["json"])
        return _Response(json.dumps(payload, ensure_ascii=False))

    monkeypatch.setattr(
        "bankreg_trustrag.llm_client.httpx.post",
        fake_post,
    )
    planner = QueryPlanner(LLMClient(LLMClientConfig(provider="openai_compatible", model="qwen", base_url="http://local/v1")))

    outcome = planner.plan("2024年9月保险业总资产是多少？")

    assert outcome.status == "ok"
    assert outcome.plan.original_query == "2024年9月保险业总资产是多少?"
    assert outcome.plan.answer_requirements[0].required_outputs == ["r1"]
    assert outcome.plan.entities.indicators == ["总资产"]
    assert outcome.plan.retrieval_tasks[0].source_scope.year == 2024
    assert outcome.plan.retrieval_tasks[0].source_scope.month == 9
    assert calls[0]["response_format"] == {"type": "json_object"}
    schema = PlannerOutput.model_json_schema()
    assert set(schema["properties"]) == {
        "user_goal",
        "answer_requirements",
        "retrieval_tasks",
        "operations",
        "requires_clarification",
    }


def test_query_planner_makes_at_most_one_json_repair_request(monkeypatch):
    responses = iter([_Response("not-json"), _Response(json.dumps(_compact_plan_payload(), ensure_ascii=False))])
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs["json"])
        return next(responses)

    monkeypatch.setattr("bankreg_trustrag.llm_client.httpx.post", fake_post)
    client = LLMClient(LLMClientConfig(provider="openai_compatible", model="planner", base_url="http://local/v1", max_retries=9))

    outcome = QueryPlanner(client).plan("2024年9月保险业总资产是多少？")

    assert outcome.status == "ok"
    assert outcome.attempts == 2
    assert len(calls) == 2
    assert all(call["response_format"] == {"type": "json_object"} for call in calls)
    assert "invalid_json" in calls[1]["messages"][-1]["content"]


def test_query_planner_canonicalizes_absolute_difference_operand_order(monkeypatch):
    monkeypatch.setattr(
        "bankreg_trustrag.llm_client.httpx.post",
        lambda *args, **kwargs: _Response(json.dumps(_cross_file_compact_payload(), ensure_ascii=False)),
    )
    planner = QueryPlanner(LLMClient(LLMClientConfig(
        provider="openai_compatible",
        model="planner",
        base_url="http://local/v1",
    )))

    outcome = planner.plan("两类公司保费合计与全国总数相差多少？")

    assert outcome.status == "ok"
    assert outcome.plan.operations[1].input_refs() == ["calc1", "r3"]
    assert outcome.plan.operations[1].parameters["absolute"] is True


def test_query_planner_repairs_omitted_subtraction_semantics(monkeypatch):
    payload = _cross_file_compact_payload()
    payload["operations"][1].pop("absolute")
    monkeypatch.setattr(
        "bankreg_trustrag.llm_client.httpx.post",
        lambda *args, **kwargs: _Response(json.dumps(payload, ensure_ascii=False)),
    )
    planner = QueryPlanner(LLMClient(LLMClientConfig(
        provider="openai_compatible",
        model="planner",
        base_url="http://local/v1",
    )))

    directional = planner.plan("商业银行合计从一季度到四季度的数值变化为多少？")
    assert directional.status == "ok"
    assert directional.plan.operations[1].parameters["absolute"] is False

    absolute = planner.plan("两类公司合计与全国总数相差多少？")
    assert absolute.status == "ok"
    assert absolute.plan.operations[1].parameters["absolute"] is True


def test_query_planner_removes_ungrounded_unit_and_completes_aggregate_column(monkeypatch):
    monkeypatch.setattr(
        "bankreg_trustrag.llm_client.httpx.post",
        lambda *args, **kwargs: _Response(json.dumps(
            _explicit_source_cross_file_payload_with_guessed_unit(), ensure_ascii=False,
        )),
    )
    planner = QueryPlanner(LLMClient(LLMClientConfig(
        provider="openai_compatible",
        model="planner",
        base_url="http://local/v1",
    )))
    question = (
        "根据《2023年10月人身险公司经营情况表》和《2023年10月财产保险公司经营情况表》，"
        "两类公司的原保险保费收入合计是多少？与《2023年10月全国各地区原保险保费收入情况表》"
        "的“全国合计”相比相差多少？"
    )

    outcome = planner.plan(question)

    assert outcome.status == "ok"
    assert [task.expected_unit for task in outcome.plan.retrieval_tasks] == [None, None, None]
    assert outcome.plan.retrieval_tasks[2].semantic_constraints.row_label == "全国合计"
    assert outcome.plan.retrieval_tasks[2].semantic_constraints.column_label == "合计"
    assert outcome.plan.operations[0].input_refs() == ["r1", "r2"]
    assert outcome.plan.operations[1].input_refs() == ["calc1", "r3"]


def test_query_planner_keeps_only_a_unit_explicitly_requested_by_user(monkeypatch):
    payload = _compact_plan_payload()
    payload["retrieval_tasks"][0]["expected_unit"] = "万元"
    monkeypatch.setattr(
        "bankreg_trustrag.llm_client.httpx.post",
        lambda *args, **kwargs: _Response(json.dumps(payload, ensure_ascii=False)),
    )
    planner = QueryPlanner(LLMClient(LLMClientConfig(
        provider="openai_compatible",
        model="planner",
        base_url="http://local/v1",
    )))

    outcome = planner.plan("请以亿元为单位查询2024年9月保险业总资产。")

    assert outcome.plan.retrieval_tasks[0].expected_unit == "亿元"


def test_query_planner_does_not_treat_attachment_as_unit(monkeypatch):
    payload = _compact_plan_payload()
    payload["retrieval_tasks"][0]["expected_unit"] = "件"
    monkeypatch.setattr(
        "bankreg_trustrag.llm_client.httpx.post",
        lambda *args, **kwargs: _Response(json.dumps(payload, ensure_ascii=False)),
    )
    planner = QueryPlanner(LLMClient(LLMClientConfig(
        provider="openai_compatible",
        model="planner",
        base_url="http://local/v1",
    )))

    outcome = planner.plan("需要对Excel附件做两处取数并计算。")

    assert outcome.status == "ok"
    assert outcome.plan.retrieval_tasks[0].expected_unit is None


def test_query_planner_fails_closed_when_llm_is_disabled():
    outcome = QueryPlanner(LLMClient(LLMClientConfig())).plan("查询总资产")

    assert outcome.plan.requires_clarification is True
    assert outcome.plan.retrieval_tasks == []


def test_llm_client_reuses_json_object_compatibility_mode(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs["json"])
        return _Response(json.dumps(_plan_payload(), ensure_ascii=False))

    monkeypatch.setattr("bankreg_trustrag.llm_client.httpx.post", fake_post)
    client = LLMClient(LLMClientConfig(provider="openai_compatible", model="qwen", base_url="http://local/v1"))
    client._json_schema_supported = False

    result = client.structured([{"role": "user", "content": "plan"}], QueryPlan, temperature=0.1)

    assert result.status == "ok"
    assert result.attempts == 1
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "必须严格符合的JSON Schema" in calls[0]["messages"][-1]["content"]


def test_planner_client_prefers_dedicated_light_model_configuration():
    settings = SimpleNamespace(
        llm_provider="openai_compatible",
        llm_model="heavy-reasoner",
        llm_base_url="http://general/v1",
        llm_api_key="general-key",
        planner_model="light-planner",
        planner_base_url="http://planner/v1",
        planner_api_key="planner-key",
        llm_planner_timeout_seconds=30,
        llm_planner_max_tokens=1800,
    )

    client = LLMClient.from_planner_settings(settings)

    assert client.config.model == "light-planner"
    assert client.config.base_url == "http://planner/v1"
    assert client.config.api_key == "planner-key"
    assert client.config.max_retries == 1
