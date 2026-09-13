from config import Settings


def test_settings_repr_hides_api_keys():
    settings = Settings(
        _env_file=None,
        EMBEDDING_API_KEY="embedding-secret",
        RERANK_API_KEY="rerank-secret",
        LLM_API_KEY="llm-secret",
        SEARCH_API_KEY="search-secret",
        VLM_API_KEY="vlm-secret",
    )

    rendered = repr(settings)

    assert "embedding-secret" not in rendered
    assert "rerank-secret" not in rendered
    assert "llm-secret" not in rendered
    assert "search-secret" not in rendered
    assert "vlm-secret" not in rendered
