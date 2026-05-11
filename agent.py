"""
agent.py - SHL Recommender Agent

"""

from __future__ import annotations

import json
import os
from dotenv import load_dotenv
load_dotenv()

import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from models import AgentState, ChatResponse, Recommendation, Constraint, Message
from catalog import Catalog, CatalogItem
from retriever import HybridRetriever
from prompts import (
    UNDERSTAND_SYSTEM, build_understand_prompt,
    CLARIFY_SYSTEM, build_clarify_prompt,
    QUERY_EXPANSION_SYSTEM, build_query_expansion_prompt,
    build_recommend_system, build_recommend_prompt,
    COMPARE_SYSTEM, build_compare_prompt,
    OFF_SCOPE_RESPONSE,
)

MAX_TURNS = 8


#  LLM 

def _get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
        max_tokens=1024,
        google_api_key=api_key,
    )


def _llm_call(system: str, user: str, temperature: float = 0.0) -> str:
    llm = _get_llm(temperature)
    result = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return result.content.strip()


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(), strict=False)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        return {}


#  Readiness gate 

def _is_ready(constraints: Constraint) -> bool:
    return any([
        constraints.role_title,
        constraints.role_description,
        constraints.seniority,
        constraints.skills_needed
    ])


#  Nodes 

def _understand(state: AgentState) -> AgentState:
    messages = [m.model_dump() for m in state.messages]
    current = state.constraints.model_dump()
    raw = _llm_call(UNDERSTAND_SYSTEM, build_understand_prompt(messages, current))
    parsed = _extract_json(raw)
    
    state.intent = parsed.get("intent", "vague")
    if "constraints" in parsed:
        c = parsed["constraints"]
        for f, v in c.items():
            if hasattr(state.constraints, f) and v:
                if isinstance(v, list):
                    setattr(state.constraints, f, list(set(getattr(state.constraints, f) + v)))
                else:
                    setattr(state.constraints, f, v)

    # Hard rules
    if state.intent in ("ready", "refine") and not _is_ready(state.constraints):
        state.intent = "vague"
    if state.turn_count >= MAX_TURNS - 1:
        state.intent = "ready"
        
    return state


def _clarify(state: AgentState) -> str:
    """Generate one targeted clarifying question. Never repeats a prior question."""
    messages_dicts = [m.model_dump() for m in state.messages]
    reply = _llm_call(
        CLARIFY_SYSTEM,
        build_clarify_prompt(messages_dicts, state.constraints.model_dump(), state.clarifying_asked),
        temperature=0.1,
    )
    state.clarifying_asked.append(reply[:120])
    return reply


def _retrieve(state: AgentState, retriever: HybridRetriever) -> AgentState:
    """
    LLM query expansion  multi-query hybrid search  rank-merge.

    The LLM generates 4-6 queries from constraints.
    Each hits ChromaDB (dense semantic) + BM25 (exact name) independently.
    Results are merged by weighted rank score - no hardcoded assessment names.
    """
    raw    = _llm_call(QUERY_EXPANSION_SYSTEM, build_query_expansion_prompt(state.constraints.model_dump()))
    parsed = _extract_json(raw)
    queries: list[str] = parsed.get("queries", [])

    if not queries:
        parts = [p for p in [
            state.constraints.role_title,
            state.constraints.seniority,
            state.constraints.purpose,
            " ".join(state.constraints.skills_needed),
            state.constraints.role_description,
        ] if p]
        queries = [" ".join(parts) or "professional assessment"]

    scores: dict[str, float]           = {}
    items_map: dict[str, CatalogItem]  = {}

    for q_rank, query in enumerate(queries[:6]):
        results = retriever.search(
            query=query,
            top_k=20,
            filter_languages=state.constraints.languages or None,
            filter_job_levels=[state.constraints.seniority] if state.constraints.seniority else None,
        )
        weight = 1.0 / (1 + q_rank * 0.15)
        for i_rank, item in enumerate(results):
            score = weight / (1 + i_rank)
            scores[item.entity_id]    = scores.get(item.entity_id, 0.0) + score
            items_map[item.entity_id] = item

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]
    state.raw_retrieval = [
        {
            "entity_id":        items_map[eid].entity_id,
            "name":             items_map[eid].name,
            "url":              items_map[eid].url,
            "primary_type":     items_map[eid].primary_type,
            "keys":             items_map[eid].keys,
            "duration_minutes": items_map[eid].duration_minutes,
            "job_levels":       items_map[eid].job_levels,
            "description":      items_map[eid].description,
        }
        for eid, _ in top
    ]
    return state


def _recommend(state: AgentState, catalog: Catalog, retriever: HybridRetriever) -> tuple[AgentState, str]:
    """LLM selects 1-10 from retrieved candidates. Output guard strips invalid URLs."""
    retrieved_items = [
        retriever.get_item(r["entity_id"]) for r in state.raw_retrieval
        if retriever.get_item(r["entity_id"])
    ]

    if not retrieved_items:
        return state, (
            "I wasn't able to find a strong match in the SHL catalog. "
            "Could you give me more detail about the role or skills to assess?"
        )

    system = build_recommend_system(retrieved_items)
    prompt = build_recommend_prompt(
        [m.model_dump() for m in state.messages],
        state.constraints.model_dump(),
    )
    raw    = _llm_call(system, prompt)
    parsed = _extract_json(raw)

    selected_ids: list[str] = parsed.get("selected_ids", [])
    reply: str              = parsed.get("reply", "Here are my recommendations:")

    recommendations: list[Recommendation] = []
    for eid in selected_ids[:10]:
        item = retriever.get_item(str(eid))
        if item and catalog.is_valid_url(item.url):
            recommendations.append(Recommendation(name=item.name, url=item.url, test_type=item.primary_type))

    if not recommendations:
        for item in retrieved_items[:5]:
            if catalog.is_valid_url(item.url):
                recommendations.append(Recommendation(name=item.name, url=item.url, test_type=item.primary_type))
        reply = "Based on your requirements, here are the most relevant assessments:"

    state.current_shortlist = recommendations
    return state, reply


def _compare(state: AgentState, catalog: Catalog, retriever: HybridRetriever) -> str:
    """Compare assessments using catalog data only. No LLM prior knowledge used."""
    latest = next((m.content for m in reversed(state.messages) if m.role == "user"), "")

    items: list[CatalogItem] = [
        item for item in catalog.all_items() if item.name.lower() in latest.lower()
    ]
    if len(items) < 2:
        seen = {i.entity_id for i in items}
        for r in retriever.search(latest, top_k=5):
            if r.entity_id not in seen:
                items.append(r)
                seen.add(r.entity_id)
            if len(items) >= 4:
                break

    return _llm_call(COMPARE_SYSTEM, build_compare_prompt([m.model_dump() for m in state.messages], items[:4]), temperature=0.1)


def _reconstruct_shortlist(state: AgentState, catalog: Catalog) -> list[Recommendation]:
    """
    Scan prior assistant messages for SHL catalog URLs already recommended.
    Returns the last complete shortlist seen.
    """
    url_pattern = re.compile(r"https://www\.shl\.com/products/product-catalog/view/[^\s\)\]>\"',]+")
    reconstructed: list[Recommendation] = []
    seen_urls: set[str] = set()

    for msg in state.messages:
        if msg.role != "assistant":
            continue
        turn_recs: list[Recommendation] = []
        for raw_url in url_pattern.findall(msg.content):
            url = raw_url.rstrip("/") + "/"
            if url in seen_urls:
                continue
            seen_urls.add(url)
            item = catalog.get_by_url(url)
            if item and catalog.is_valid_url(item.url):
                turn_recs.append(Recommendation(name=item.name, url=item.url, test_type=item.primary_type))
        if turn_recs:
            reconstructed = turn_recs  # keep the last non-empty turn

    return reconstructed


#  Agent 

class SHLAgent:

    def __init__(self, catalog: Catalog, retriever: HybridRetriever):
        self.catalog   = catalog
        self.retriever = retriever

    def run(self, state: AgentState) -> ChatResponse:
        state = _understand(state)
        intent = state.intent
        
        reply, recs, eoc = "", [], False
        
        if intent == "off_scope":
            reply = OFF_SCOPE_RESPONSE
        elif intent == "satisfied":
            recs = _reconstruct_shortlist(state, self.catalog)
            if not recs:
                state = _retrieve(state, self.retriever)
                state, _ = _recommend(state, self.catalog, self.retriever)
                recs = state.current_shortlist
            reply = "Glad I could help!"
            eoc = True
        elif intent == "compare":
            reply = _compare(state, self.catalog, self.retriever)
        elif intent in ("ready", "refine"):
            state = _retrieve(state, self.retriever)
            state, reply = _recommend(state, self.catalog, self.retriever)
            recs = state.current_shortlist
        else:
            reply = _clarify(state)

        return ChatResponse(
            reply=reply,
            recommendations=[r for r in recs if self.catalog.is_valid_url(r.url)],
            end_of_conversation=eoc
        )
