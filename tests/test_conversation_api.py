from fastapi.testclient import TestClient

from bankreg_trustrag.api import create_app
from bankreg_trustrag.config import Settings
from bankreg_trustrag.storage import Store


def _settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "data",
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "bankreg.sqlite3",
        bge_mode="disabled",
        bge_vector_dir=tmp_path / "vectors",
    )


def test_conversation_messages_are_isolated_by_memory_scope(tmp_path):
    store = Store(tmp_path / "memory.sqlite3")
    store.create_conversation("conv_one", "browser_scope_one")
    store.add_conversation_message("msg_one", "conv_one", "browser_scope_one", "user", "资本充足率要求是什么？")

    assert store.conversation_messages("conv_one", "browser_scope_one")[0]["content"] == "资本充足率要求是什么？"
    assert store.conversation_messages("conv_one", "browser_scope_two") == []


def test_long_term_memory_recall_only_returns_matching_answered_exchange(tmp_path):
    store = Store(tmp_path / "memory.sqlite3")
    store.create_conversation("conv_one", "browser_scope_one")
    store.remember_answer(
        "mem_one",
        "browser_scope_one",
        "conv_one",
        "不良贷款率是多少？",
        "不良贷款率为 1.2%。",
        "table_lookup",
        "answer",
        ["cell:report:Sheet1:C8"],
    )
    store.remember_answer(
        "mem_two",
        "browser_scope_one",
        "conv_one",
        "资本充足率是多少？",
        "资本充足率为 10%。",
        "table_lookup",
        "clarify",
        [],
    )

    recalled = store.recall_memories("browser_scope_one", "不良贷款率的监管口径")

    assert [item["memory_id"] for item in recalled] == ["mem_one"]


def test_delete_conversation_removes_messages_memories_and_chat_audit(tmp_path):
    store = Store(tmp_path / "memory.sqlite3")
    store.create_conversation("conv_delete", "browser_scope_one")
    store.add_conversation_message(
        "msg_delete", "conv_delete", "browser_scope_one", "assistant", "已回答",
        trace_id="trace_delete",
    )
    store.remember_answer(
        "mem_delete", "browser_scope_one", "conv_delete", "问题", "答案",
        "regulatory_fact", "answer", [],
    )
    store.save_qa(
        "trace_delete", "问题", "regulatory_fact", {}, [], "答案", 0.9,
        {}, "answer", 10,
    )

    assert store.delete_conversation("conv_delete", "browser_scope_one") is True
    assert store.get_conversation("conv_delete", "browser_scope_one") is None
    assert store.conversation_messages("conv_delete", "browser_scope_one") == []
    assert store.recall_memories("browser_scope_one", "问题") == []
    assert store.history() == []
    assert store.delete_conversation("conv_delete", "browser_scope_one") is False


def test_delete_conversation_cannot_cross_memory_scope(tmp_path):
    store = Store(tmp_path / "memory.sqlite3")
    store.create_conversation("conv_scope", "browser_scope_one")

    assert store.delete_conversation("conv_scope", "browser_scope_two") is False
    assert store.get_conversation("conv_scope", "browser_scope_one") is not None


def test_delete_conversation_endpoint_is_scope_protected(tmp_path):
    client = TestClient(create_app(_settings(tmp_path)))
    created = client.post(
        "/api/conversations",
        json={"memory_scope_id": "browser_api_scope", "title": "待删除"},
    )
    assert created.status_code == 200
    conversation_id = created.json()["conversation_id"]

    blocked = client.delete(
        f"/api/conversations/{conversation_id}?memory_scope_id=browser_other_scope",
    )
    assert blocked.status_code == 404

    deleted = client.delete(
        f"/api/conversations/{conversation_id}?memory_scope_id=browser_api_scope",
    )
    assert deleted.status_code == 204
    assert client.get("/api/conversations?memory_scope_id=browser_api_scope").json() == []


def test_chat_stream_emits_public_workflow_status_and_answer(tmp_path):
    app = create_app(_settings(tmp_path))
    response = TestClient(app).post(
        "/api/chat/stream",
        json={
            "question": "商业银行资本充足率要求是什么？",
            "memory_scope_id": "browser_stream_test",
        },
    )

    assert response.status_code == 200
    assert "event: conversation" in response.text
    assert "event: status" in response.text
    assert "event: answer_delta" in response.text
    assert "event: complete" in response.text
