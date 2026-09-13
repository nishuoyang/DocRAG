from config import get_settings
from core import retrieval


def test_query_plan_uses_hyde_only_for_dense_retrieval(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "RETRIEVAL_MODE", "quality")
    monkeypatch.setattr(settings, "QUERY_TRANSFORM", True)
    monkeypatch.setattr(settings, "HYDE", True)
    monkeypatch.setattr(retrieval.query_transform, "transform_query", lambda q, history=None: "改写问题")
    monkeypatch.setattr(retrieval.query_transform, "hyde_query", lambda q, history=None: "假想答案")

    plan = retrieval._build_query_plan("原始问题", history=[{"role": "user", "content": "历史"}])

    assert plan.dense == ["原始问题", "改写问题", "假想答案"]
    assert plan.sparse == ["原始问题", "改写问题"]


def test_balanced_mode_skips_enhancement_for_standalone_question(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "RETRIEVAL_MODE", "balanced")
    monkeypatch.setattr(settings, "QUERY_TRANSFORM", True)
    monkeypatch.setattr(settings, "HYDE", True)
    monkeypatch.setattr(
        retrieval.query_transform,
        "transform_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not transform")),
    )
    monkeypatch.setattr(
        retrieval.query_transform,
        "hyde_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call hyde")),
    )

    plan = retrieval._build_query_plan("github发布项目的完整流程是什么")

    assert plan.dense == ["github发布项目的完整流程是什么"]
    assert plan.sparse == ["github发布项目的完整流程是什么"]


def test_fast_mode_skips_all_query_enhancement(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "RETRIEVAL_MODE", "fast")
    monkeypatch.setattr(settings, "QUERY_TRANSFORM", True)
    monkeypatch.setattr(settings, "HYDE", True)

    plan = retrieval._build_query_plan("它的流程呢")

    assert plan.dense == ["它的流程呢"]
    assert plan.sparse == ["它的流程呢"]


def test_query_plan_skips_llm_enhancement_for_exact_identifiers(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "RETRIEVAL_MODE", "quality")
    monkeypatch.setattr(settings, "QUERY_TRANSFORM", True)
    monkeypatch.setattr(settings, "HYDE", True)
    monkeypatch.setattr(
        retrieval.query_transform,
        "transform_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not transform")),
    )
    monkeypatch.setattr(
        retrieval.query_transform,
        "hyde_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call hyde")),
    )

    plan = retrieval._build_query_plan("PT-001 的适用场景是什么？")

    assert plan.dense == ["PT-001 的适用场景是什么？"]
    assert plan.sparse == ["PT-001 的适用场景是什么？"]
