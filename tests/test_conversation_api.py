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
