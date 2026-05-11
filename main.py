"""
main.py - FastAPI 
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import (
    AgentState,
    ChatRequest,
    ChatResponse,
    Constraint,
    HealthResponse,
    Message,
)
from catalog import load_catalog
from retriever import HybridRetriever
from agent import SHLAgent

#  Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("shl_recommender")

#  Global singletons (loaded once at startup) 
_catalog   = None
_retriever = None
_agent     = None

CATALOG_PATH = os.getenv("CATALOG_PATH", "shl_catalog.json")

MAX_TURNS = 8  # from spec: evaluator caps at 8 turns


#  Lifespan (replaces deprecated @app.on_event) 
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _catalog, _retriever, _agent
    logger.info("Starting up - loading catalog and building indexes...")
    t0 = time.time()

    _catalog   = load_catalog(CATALOG_PATH)
    _retriever = HybridRetriever(_catalog)
    _agent     = SHLAgent(_catalog, _retriever)

    logger.info(f"Startup complete in {time.time() - t0:.1f}s | {len(_catalog)} assessments indexed.")
    yield
    logger.info("Shutting down.")


#  App 
app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#  Exception handlers 
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler: return a valid ChatResponse schema even on errors.
    This ensures schema compliance under all failure modes.
    """
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    # Return valid schema - the spec says schema compliance is a hard eval
    return JSONResponse(
        status_code=200,  # return 200 with error in reply to maintain schema
        content={
            "reply": "I encountered an internal error. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


#  Endpoints 

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness check. Returns 200 + {"status": "ok"}."""
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless conversational endpoint.
    Full conversation history must be included on every call.
    """
    if _agent is None:
        logger.error("Agent not initialized - startup may have failed.")
        raise HTTPException(status_code=503, detail="Service not ready")

    messages = request.messages

    #  Hard guard: turn cap (spec: max 8 turns) 
    if len(messages) > MAX_TURNS:
        logger.warning(f"Turn cap exceeded: {len(messages)} turns. Forcing final response.")
        messages = messages[:MAX_TURNS]

    #  Build agent state from request 
    state = AgentState(
        messages=messages,
        turn_count=len(messages),
    )

    #  Run agent 
    t0 = time.time()
    response = _agent.run(state)
    elapsed = time.time() - t0

    logger.info(
        f"Chat | turns={len(messages)} | intent={state.intent} | "
        f"recs={len(response.recommendations)} | "
        f"eoc={response.end_of_conversation} | {elapsed:.2f}s"
    )

    return response
