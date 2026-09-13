"""Helpers for compact, structured LangSmith traces."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.documents import Document


def _preview(value: Any, limit: int = 240) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[:limit]}..."


def document_summary(document: Document) -> dict:
    metadata = document.metadata
    return {
        "filename": metadata.get("filename"),
        "section": metadata.get("section"),
        "chunk_index": metadata.get("chunk_index"),
        "matched_chunk_index": metadata.get("matched_chunk_index"),
        "parent_id": metadata.get("parent_id"),
        "child_index": metadata.get("child_index"),
        "page_start": metadata.get("page_start", metadata.get("page")),
        "page_end": metadata.get("page_end", metadata.get("page")),
        "rrf_score": metadata.get("rrf_score"),
        "relevance_score": metadata.get("relevance_score"),
        "content_chars": len(document.page_content),
        "content_preview": _preview(document.page_content),
    }


def document_list_summary(documents: list[Document] | None) -> dict:
    items = documents or []
    return {
        "count": len(items),
        "documents": [document_summary(document) for document in items],
    }


def process_retrieval_inputs(inputs: dict) -> dict:
    history = inputs.get("history") or []
    return {
        "query": inputs.get("query"),
        "top_k": inputs.get("top_k"),
        "history_turns": len(history),
    }


def process_document_outputs(outputs: Any) -> dict:
    if isinstance(outputs, list) and all(isinstance(item, Document) for item in outputs):
        return document_list_summary(outputs)
    return {"output": outputs}


def process_document_inputs(inputs: dict) -> dict:
    documents = inputs.get("docs") or inputs.get("ranked") or []
    return {"document_count": len(documents)}


def process_query_plan_outputs(outputs: Any) -> dict:
    return {
        "dense_queries": list(getattr(outputs, "dense", []) or []),
        "sparse_queries": list(getattr(outputs, "sparse", []) or []),
    }


def process_query_inputs(inputs: dict) -> dict:
    history = inputs.get("history") or []
    return {
        "query": inputs.get("query"),
        "history_turns": len(history),
    }


def process_text_output(output: Any) -> dict:
    return {"query": _preview(output, 800)}


def process_context_output(output: Any) -> dict:
    text = str(output or "")
    return {
        "context_chars": len(text),
        "context_preview": _preview(text, 800),
    }


def process_message_inputs(inputs: dict) -> dict:
    messages = inputs.get("messages") or []
    last_user = ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user = str(message.get("content") or "")
            break
        if getattr(message, "type", None) == "human":
            last_user = str(getattr(message, "content", "") or "")
            break
    return {
        "message_count": len(messages),
        "last_user_preview": _preview(last_user),
    }


def process_answer_output(output: Any) -> dict:
    content = getattr(output, "content", output)
    return {"answer_preview": _preview(content, 1200)}


def join_text_chunks(chunks: list[str]) -> dict:
    answer = "".join(str(chunk) for chunk in chunks if chunk)
    return {
        "answer": answer,
        "answer_chars": len(answer),
    }


def process_candidate_inputs(inputs: dict) -> dict:
    result_lists = inputs.get("result_lists") or []
    return {
        "result_list_count": len(result_lists),
        "candidate_counts": [len(items) for items in result_lists],
    }


def process_search_inputs(inputs: dict) -> dict:
    return {
        "query": inputs.get("query"),
        "k": inputs.get("k"),
    }


def process_batch_search_inputs(inputs: dict) -> dict:
    queries = inputs.get("queries") or []
    return {
        "query_count": len(queries),
        "queries": list(queries),
        "k": inputs.get("k"),
    }


def process_document_lists_outputs(outputs: Any) -> dict:
    result_lists = outputs or []
    return {
        "result_list_count": len(result_lists),
        "document_counts": [len(items) for items in result_lists],
    }


def process_rerank_inputs(inputs: dict) -> dict:
    return {
        "query": inputs.get("query"),
        "top_n": inputs.get("top_n"),
        "candidate_count": len(inputs.get("candidates") or []),
    }


def process_parent_inputs(inputs: dict) -> dict:
    return {
        "ranked_count": len(inputs.get("ranked") or []),
        "max_parents": inputs.get("max_parents"),
        "max_chars": inputs.get("max_chars"),
    }


def process_source_outputs(outputs: Any) -> dict:
    if not isinstance(outputs, dict):
        return {"output": str(outputs)[:1000]}
    sources = outputs.get("sources") or []
    return {
        "answer_preview": _preview(outputs.get("answer"), 1200),
        "source_count": len(sources),
        "sources": [
            {
                "citation_index": source.get("citation_index"),
                "filename": source.get("filename"),
                "url": source.get("url"),
                "section": source.get("section"),
                "page_start": source.get("page_start"),
                "page_end": source.get("page_end"),
            }
            for source in sources
        ],
    }


def process_citation_inputs(inputs: dict) -> dict:
    answer = inputs.get("answer") or ""
    return {
        "answer_chars": len(answer),
        "document_count": len(inputs.get("docs") or []),
    }


def process_citation_outputs(outputs: Any) -> dict:
    citations = []
    for citation_index, document in outputs or []:
        citations.append(
            {
                "citation_index": citation_index,
                **document_summary(document),
            }
        )
    return {"count": len(citations), "citations": citations}


def reduce_sse_events(events: list[str]) -> dict:
    """Reduce streamed SSE strings into a compact LangSmith output."""
    answer = ""
    sources: list[dict] = []
    for event in events:
        if not isinstance(event, str) or not event.startswith("data: "):
            continue
        payload = event[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            if data.get("delta"):
                answer += str(data["delta"])
            elif data.get("answer"):
                answer = str(data["answer"])
            elif data.get("sources"):
                sources = data["sources"]
    return {
        "answer": answer,
        "source_count": len(sources),
        "sources": [
            {
                "citation_index": source.get("citation_index"),
                "filename": source.get("filename"),
                "page_start": source.get("page_start"),
                "page_end": source.get("page_end"),
            }
            for source in sources
        ],
    }
