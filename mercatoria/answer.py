"""Optional answer generation over retrieved chunks.

Generation is kept outside the retriever so an agent can call ``/retrieve`` and
use its own model. When enabled, this module calls an OpenAI-compatible chat
endpoint and requires citations in the form ``[chunk_id]``.
"""

from __future__ import annotations

import re
from typing import Any

from .config import AnswerConfig
from .http_client import HttpError, JsonHttp

CITATION_RE = re.compile(r"\[([0-9a-f]{16})\]")
REFUSAL_MARKERS = (
    "依据不足",
    "没有相关",
    "未找到",
    "无法回答",
    "not found",
    "insufficient",
    "cannot answer",
    "no relevant",
)


class ChatError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, cfg: AnswerConfig):
        self.cfg = cfg
        self._http = JsonHttp(cfg.base, cfg.timeout)

    def chat(self, messages: list[dict[str, str]]) -> str:
        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = "Bearer " + self.cfg.api_key
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        try:
            payload = self._http.post("chat/completions", body, headers)
            return str(payload["choices"][0]["message"]["content"] or "")
        except (HttpError, RuntimeError, TimeoutError, KeyError, TypeError) as exc:
            raise ChatError(f"chat request failed: {exc}") from exc


def _prompt(
    question: str,
    chunks: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    blocks = []
    for chunk in chunks:
        cid = chunk["chunk_id"]
        label = chunk.get("article_ids") or chunk.get("articles") or []
        label_text = ", ".join(str(x) for x in label)
        source = chunk.get("source_doc", "")
        text = str(chunk.get("text", ""))[:4000]
        blocks.append(f"[{cid}] ({source} / {label_text})\n{text}")
    system = (
        "你是进出口贸易法规检索助手。只允许使用用户消息中给出的条文回答。"
        "每条结论后必须用 [chunk_id] 标注来源；chunk_id 必须逐字来自条文列表。"
        "如果条文不足以回答问题，明确写“依据不足”，并说明缺少哪类规则。"
        "不要编造条文、条号或来源。"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    # Keep the last four turns of the conversation. Prior assistant replies may
    # carry old chunk ids; the system prompt constrains citations to the
    # references supplied in the final user message.
    for turn in (history or [])[-8:]:
        role = turn.get("role")
        content = str(turn.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content[:4000]})
    user = "问题：\n" + question + "\n\n可用条文：\n\n" + "\n\n---\n\n".join(blocks)
    messages.append({"role": "user", "content": user})
    return messages


def _citations(answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {c["chunk_id"]: c for c in chunks}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cid in CITATION_RE.findall(answer):
        if cid in seen or cid not in by_id:
            continue
        seen.add(cid)
        chunk = by_id[cid]
        out.append(
            {
                "chunk_id": cid,
                "doc_id": chunk.get("doc_id"),
                "article_id": (chunk.get("article_ids") or [None])[0],
                "article_label": (chunk.get("articles") or [None])[0],
                "source_doc": chunk.get("source_doc"),
                "quote": str(chunk.get("text", ""))[:500],
                "text": str(chunk.get("text", "")),
            }
        )
    return out


def _references(
    chunks: list[dict[str, Any]], citations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Full original passages used for the answer, in retrieval rank order."""
    cited = {c["chunk_id"] for c in citations}
    refs: list[dict[str, Any]] = []
    for chunk in chunks:
        score_parts = chunk.get("score_parts") or {}
        refs.append(
            {
                "chunk_id": chunk["chunk_id"],
                "rank": chunk.get("rank"),
                "doc_id": chunk.get("doc_id"),
                "source_doc": chunk.get("source_doc"),
                "article_id": (chunk.get("article_ids") or [None])[0],
                "article_label": (chunk.get("articles") or [None])[0],
                "jurisdiction": (chunk.get("jurisdictions") or [None])[0],
                "effective_from": chunk.get("effective_from"),
                "effective_to": chunk.get("effective_to"),
                "score": chunk.get("score"),
                "rerank_score": score_parts.get("rerank"),
                "cited": chunk["chunk_id"] in cited,
                "text": str(chunk.get("text", "")),
                "article_offsets": chunk.get("article_offsets"),
            }
        )
    return refs


def answer_question(
    cfg: AnswerConfig,
    question: str,
    chunks: list[dict[str, Any]],
    client: ChatClient | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if not cfg.enabled:
        return {
            "answer": None,
            "answer_status": "skipped",
            "citations": [],
            "references": _references(chunks, []),
            "error": None,
        }
    if not chunks:
        return {
            "answer": None,
            "answer_status": "unavailable",
            "citations": [],
            "references": [],
            "error": None,
            "message": "no retrieved chunks",
        }
    chunks = chunks[: max(1, cfg.max_context_chunks)]
    try:
        text = (client or ChatClient(cfg)).chat(_prompt(question, chunks, history))
    except ChatError as exc:
        return {
            "answer": None,
            "answer_status": "error",
            "citations": [],
            "references": _references(chunks, []),
            "error": str(exc),
        }

    citations = _citations(text, chunks)
    lowered = text.lower()
    refused = any(marker in lowered for marker in REFUSAL_MARKERS)
    if not text.strip():
        status = "error"
    elif refused and not citations:
        status = "unavailable"
    elif refused:
        status = "partial"
    elif not citations:
        status = "unsupported"
    else:
        status = "answered"
    return {
        "answer": text,
        "answer_status": status,
        "citations": citations,
        "references": _references(chunks, citations),
        "error": None,
    }
