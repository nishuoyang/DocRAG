from core import query_transform


def test_transform_query_uses_recent_history_for_pronoun_resolution(monkeypatch):
    captured = {}

    def fake_complete(prompt: str) -> str:
        captured["prompt"] = prompt
        return "BGE-M3 的价格是多少？"

    monkeypatch.setattr(query_transform, "_llm_complete", fake_complete)

    result = query_transform.transform_query(
        "它的价格呢？",
        history=[
            {"role": "user", "content": "介绍一下 BGE-M3"},
            {"role": "assistant", "content": "BGE-M3 是向量模型。"},
        ],
    )

    assert result == "BGE-M3 的价格是多少？"
    assert "介绍一下 BGE-M3" in captured["prompt"]
    assert "BGE-M3 是向量模型" in captured["prompt"]
