from types import SimpleNamespace

from bankreg_trustrag.config import Settings
from bankreg_trustrag.generation import GenerationResult, GroundedGenerator, LLMConfig, clean_user_answer
from bankreg_trustrag.retrieval.index import Hit
from bankreg_trustrag.schemas import ParsedQuery
from bankreg_trustrag.service import TrustRAGService


def _hit(content="商业银行不得挪用客户资金。"):
    return Hit(
        "text",
        {
            "evidence_id": "text:d1:p1",
            "doc_id": "d1",
            "content": content,
            "source_title": "资金管理办法",
            "source_file_name": "rule.docx",
            "page": 2,
            "paragraph_no": 3,
            "source_local_path": "secret/local/path.docx",
        },
        fused_score=0.2,
    )


def test_generator_builds_context_from_selected_evidence_without_local_path():
    generator = GroundedGenerator(LLMConfig(provider="openai_compatible", model="qwen", base_url="http://127.0.0.1:9000/v1"))

    context, evidence_ids = generator.build_context([_hit()])

    assert evidence_ids == ["text:d1:p1"]
    assert "商业银行不得挪用客户资金" in context
    assert "text:d1:p1" not in context
    assert "secret/local/path.docx" not in context


def test_settings_accepts_deepseek_environment_aliases(monkeypatch, tmp_path):
    monkeypatch.delenv("BANKREG_LLM_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "alias-key")
    monkeypatch.setenv("BANKREG_LLM_TIMEOUT", "120")
    monkeypatch.setenv("BANKREG_CONTEXT_MAX_CHARS", "30000")

    settings = Settings.from_env(tmp_path)

    assert settings.llm_api_key == "alias-key"
    assert settings.llm_timeout_seconds == 120
    assert settings.llm_max_context_chars == 30000


def test_generator_sends_retrieved_context_to_chat_endpoint(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "商业银行不得挪用客户资金。[证据: text:d1:p1]"}}]}

    def fake_post(url, **kwargs):
        request["url"] = url
        request.update(kwargs)
        return Response()

    monkeypatch.setattr("bankreg_trustrag.generation.httpx.post", fake_post)
    generator = GroundedGenerator(LLMConfig(provider="openai_compatible", model="qwen", base_url="http://127.0.0.1:9000/v1"))
    parsed = ParsedQuery("资金使用要求是什么", "regulatory_fact")

    result = generator.generate("资金使用要求是什么", parsed, [_hit()], [])

    assert result.status == "generated"
    assert request["url"] == "http://127.0.0.1:9000/v1/chat/completions"
    assert "text:d1:p1" not in request["json"]["messages"][1]["content"]
    assert "商业银行不得挪用客户资金" in request["json"]["messages"][1]["content"]
    system_prompt = request["json"]["messages"][0]["content"]
    user_prompt = request["json"]["messages"][1]["content"]
    assert "最小充分证据" in system_prompt
    assert "明确反证" in system_prompt
    assert "规则文件提供的规则、数据文件提供的数据、比较过程与最终结论" in system_prompt
    assert "INTENT: lookup" in user_prompt
    assert "ANSWER_FORMAT: free_text" in user_prompt
    assert "REQUIREMENTS:" in user_prompt
    assert result.answer == "商业银行不得挪用客户资金。"


def test_clean_user_answer_removes_internal_evidence_markers():
    answer = "一季度：100亿元 [证据: cell:DOC_123:Sheet1:C5]\n二季度：101亿元"

    assert clean_user_answer(answer) == "一季度：100亿元\n二季度：101亿元"


def test_service_accepts_only_verified_llm_answer():
    class FakeGenerator:
        enabled = True
        config = SimpleNamespace(provider="fake", model="test-model")

        def status(self):
            return {"provider": "fake", "model": "test-model", "enabled": True}

        def generate(self, question, parsed, hits, operations):
            assert hits[0].evidence_id == "text:d1:p1"
            return GenerationResult("generated", "商业银行不得挪用客户资金。[证据: text:d1:p1]", ("text:d1:p1",))

    class FakeIndex:
        doc_by_id = {"d1": {"title": "资金管理办法", "file_name": "rule.docx", "status": "effective"}}
        model_status = {"mode": "disabled"}

        def hybrid_search(self, *args, **kwargs):
            return [_hit()]

    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(min_trust=0.58, top_k=8)
    service.store = SimpleNamespace(save_qa=lambda *args: None)
    service.index = FakeIndex()
    service.semantic = SimpleNamespace(enabled=False)
    service.generator = FakeGenerator()

    response = service.ask("商业银行资金使用有什么要求？")

    assert response.answer == "商业银行不得挪用客户资金。[证据: text:d1:p1]"
    assert response.query_plan["generation"]["status"] == "accepted"
    assert response.query_plan["agent_workflow"]["answer_generation"]["strategy"] == "llm_grounded_with_deterministic_fallback"
