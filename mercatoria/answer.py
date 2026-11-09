"""Optional answer generation over retrieved chunks.

Generation is kept outside the retriever so an agent can call ``/retrieve`` and
use its own model. When enabled, this module calls an OpenAI-compatible chat
endpoint and requires citations in the form ``[chunk_id]``.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from .config import AnswerConfig

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

    def chat(self, messages: list[dict[str, str]]) -> str:
        url = self.cfg.base.rstrip("/") + "/chat/completions"
        body = json.dumps(
            {
                "model": self.cfg.model,
                "messages": messages,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = "Bearer " + self.cfg.api_key
        req = urllib.request.Request(url, body, headers)
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return str(payload["choices"][0]["message"]["content"] or "")
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            raise ChatError(f"chat request failed: {exc}") from exc


def _prompt(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
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
    user = "问题：\n" + question + "\n\n可用条文：\n\n" + "\n\n---\n\n".join(blocks)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
            }
        )
    return out


def answer_question(
    cfg: AnswerConfig,
    question: str,
    chunks: list[dict[str, Any]],
    client: ChatClient | None = None,
) -> dict[str, Any]:
    if not cfg.enabled:
        return {"answer": None, "answer_status": "skipped", "citations": [], "error": None}
    if not chunks:
        return {
            "answer": None,
            "answer_status": "unavailable",
            "citations": [],
            "error": None,
            "message": "no retrieved chunks",
        }
    chunks = chunks[: max(1, cfg.max_context_chunks)]
    try:
        text = (client or ChatClient(cfg)).chat(_prompt(question, chunks))
    except ChatError as exc:
        return {
            "answer": None,
            "answer_status": "error",
            "citations": [],
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
        "error": None,
    }
