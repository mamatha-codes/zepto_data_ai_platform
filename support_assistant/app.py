import os
from typing import TypedDict

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from support_assistant.vector_store import retrieve_policy


# ============================================================
# Configuration
# ============================================================

MOCK_LLM = os.getenv("MOCK_LLM", "1")


# ============================================================
# Prompt Skeleton
# ============================================================

PROMPT_SKELETON = """
Role:
You are a Zepto customer-support policy assistant.

Context:
Use only the policy context retrieved from the local policy documents.

Task:
Answer the customer's question using the provided policy context.

Format:
Return a concise and helpful answer.

Length:
Keep the answer short and easy to understand.

Negative constraint:
Do not invent policy details that are not present in the provided context.

Few-shot example:
Question: What should I do if my order is delayed?
Answer: Please check the latest tracking information in the app. If the tracking information has not changed for an extended period, contact support with your order details.
"""


# ============================================================
# Pydantic Output Schema
# ============================================================

class AssistantResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# ============================================================
# LangGraph State
# ============================================================

class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_context: list[dict]
    answer: str
    sources: list[str]
    confidence: float


# ============================================================
# Intent Classification
# ============================================================

POLICY_KEYWORDS = [
    "delivery",
    "delay",
    "delayed",
    "return",
    "refund",
    "membership",
    "tracking",
    "track",
    "cancel",
    "cancellation",
    "gift card",
    "giftcard",
    "support",
    "support hours",
    "payment",
    "charged",
]


def classify_intent(state: AssistantState):
    """Classify the user query using a deterministic mock classifier."""

    query = state["query"].lower()

    if any(keyword in query for keyword in POLICY_KEYWORDS):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        "intent": intent
    }


# ============================================================
# Retrieve + Answer
# ============================================================

def retrieve_and_answer(state: AssistantState):
    """Retrieve top-3 policy chunks and generate a deterministic answer."""

    query = state["query"]

    results = retrieve_policy(
        query=query,
        top_k=3,
    )

    if results:
        top_chunk = results[0]["text"]

        answer = (
            f"Based on the retrieved context: "
            f"{top_chunk}"
        )

        sources = [
            result["source"]
            for result in results
        ]

        confidence = 0.90

    else:
        answer = (
            "I could not find relevant policy information "
            "for that question."
        )

        sources = []
        confidence = 0.20

    return {
        "retrieved_context": results,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


# ============================================================
# Direct Answer
# ============================================================

def direct_answer(state: AssistantState):
    """Handle questions outside the supported policy scope."""

    answer = (
        "I can only answer questions about Zepto policies right now."
    )

    return {
        "answer": answer,
        "sources": [],
        "confidence": 0.95,
    }


# ============================================================
# Conditional Routing
# ============================================================

def route_by_intent(state: AssistantState):
    """Route policy questions to retrieval and other questions to direct answer."""

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# Build LangGraph
# ============================================================

graph_builder = StateGraph(AssistantState)

graph_builder.add_node(
    "classify_intent",
    classify_intent,
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer,
)

graph_builder.add_node(
    "direct_answer",
    direct_answer,
)

graph_builder.set_entry_point("classify_intent")

graph_builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

graph_builder.add_edge(
    "retrieve_and_answer",
    END,
)

graph_builder.add_edge(
    "direct_answer",
    END,
)

graph = graph_builder.compile()


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description="Policy-based Zepto customer support assistant",
    version="1.0.0",
)


class AskRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "Zepto Support Assistant is running."
    }


@app.post(
    "/ask",
    response_model=AssistantResponse,
)
def ask(request: AskRequest):

    result = graph.invoke(
        {
            "query": request.query
        }
    )

    response = AssistantResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
    )

    return response