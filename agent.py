"""
agent.py
--------
The Corrective RAG (CRAG) agent, implemented as a LangGraph state machine.

Flow (matches the project spec):

    START
      -> retrieve                 (Node 1: pull top-K chunks from Pinecone)
      -> grade_documents           (Node 2: LLM-as-judge relevance grading)
           -- relevant docs found --------> generate
           -- no relevant docs -----------> rewrite_query
      -> rewrite_query             (Node 3: rewrite question for web search)
      -> web_search                (Node 4: Tavily fallback, appends context)
      -> generate                  (Node 5: answer strictly from context)
      -> [hallucination check]     (Node 6: is the answer grounded?)
           -- grounded ------------------> END
           -- not grounded, retries left -> regenerate (bounded loop)
           -- not grounded, retries used -> fallback (safe failure message)

The bounded retry loop on Node 6 is a small, deliberate addition beyond the
literal spec: an ungrounded-generation loop with no cap is a real production
failure mode (infinite loop / runaway API cost), so we cap self-correction
attempts and degrade to an honest "I don't know" rather than looping forever.
"""

from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import MAX_GENERATION_RETRIES, TOP_K, get_llm, get_or_create_index, get_vectorstore


# ==========================================================================
# STATE
# ==========================================================================
class GraphState(TypedDict):
    """Shared state threaded through every node in the graph."""

    original_question: str      # the user's question, verbatim -- always
                                 # used for final answer generation so the
                                 # answer stays on-topic even after rewrites
    question: str                # working question, may be rewritten for search
    generation: str               # the LLM's current answer draft
    documents: List[Document]     # current working context (Pinecone + web)
    web_search_needed: str        # "Yes" / "No", set by the document grader
    generation_retries: int       # hallucination self-correction loop counter


# ==========================================================================
# STRUCTURED GRADER SCHEMAS
# ==========================================================================
class GradeDocuments(BaseModel):
    """Binary relevance verdict for a single retrieved chunk."""

    binary_score: str = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary groundedness verdict for a generated answer."""

    binary_score: str = Field(
        description="Answer is grounded in / supported by the given facts, 'yes' or 'no'"
    )


# ==========================================================================
# PROMPTS
# ==========================================================================
GRADE_DOCUMENTS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing the relevance of a retrieved document chunk "
            "to a user question about SEC 10-K filings. This is a strict but not "
            "brittle check: if the chunk contains keywords or semantic meaning "
            "related to the question, grade it as relevant. Give a binary score "
            "'yes' or 'no' to indicate whether the chunk is relevant to the question.",
        ),
        ("human", "Retrieved chunk:\n\n{document}\n\nUser question: {question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query re-writer that converts an input financial question "
            "into a better-optimized version for a live web search. Look at the "
            "input and reason about the underlying intent. Preserve company names, "
            "fiscal years, and specific metrics. Return ONLY the rewritten query, "
            "with no preamble or explanation.",
        ),
        ("human", "Here is the initial question:\n\n{question}\n\nFormulate an improved web search query."),
    ]
)

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise financial analyst assistant. Answer the user's "
            "question using ONLY the provided context, which may include excerpts "
            "from SEC 10-K filings and/or live web search results. Cite whether "
            "each fact you use comes from 'the filing' or 'a web search' when it "
            "materially affects the answer. If the context does not contain "
            "enough information to answer confidently, explicitly say so instead "
            "of guessing. Be concise and quantitative where the context supports it.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether an LLM-generated answer is "
            "grounded in / fully supported by a given set of facts. Give a binary "
            "score 'yes' or 'no'. 'Yes' means every material claim in the answer "
            "is backed by the facts; 'no' means the answer includes claims not "
            "present in or contradicted by the facts.",
        ),
        ("human", "Set of facts:\n\n{documents}\n\nLLM generation:\n\n{generation}"),
    ]
)


def _format_docs(docs: List[Document]) -> str:
    """Render documents with their source tagged, so the generator (and the
    hallucination grader) can distinguish filing content from web content."""
    if not docs:
        return "No context was retrieved."
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )


# ==========================================================================
# NODES
# ==========================================================================
def retrieve(state: GraphState) -> dict:
    """Node 1: fetch top-K chunks from Pinecone for the current question."""
    index = get_or_create_index()
    vectorstore = get_vectorstore(index)
    docs = vectorstore.similarity_search(state["question"], k=TOP_K)
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict:
    """Node 2: LLM-as-judge relevance grading of each retrieved chunk.

    Chunks graded irrelevant are dropped from the working context. If NONE
    of the retrieved chunks are relevant, we flag that a web search fallback
    is needed; otherwise we proceed to generation with whatever relevant
    subset remains (classic Corrective-RAG behavior).
    """
    grader = GRADE_DOCUMENTS_PROMPT | get_llm(temperature=0).with_structured_output(GradeDocuments)

    filtered_docs = []
    for doc in state["documents"]:
        verdict = grader.invoke({"question": state["question"], "document": doc.page_content})
        if verdict.binary_score.strip().lower() == "yes":
            filtered_docs.append(doc)

    web_search_needed = "Yes" if len(filtered_docs) == 0 else "No"
    return {"documents": filtered_docs, "web_search_needed": web_search_needed}


def decide_to_generate(state: GraphState) -> str:
    """Conditional edge after grading: route to web search fallback or straight to generation."""
    return "rewrite_query" if state["web_search_needed"] == "Yes" else "generate"


def rewrite_query(state: GraphState) -> dict:
    """Node 3: rewrite the question into a better web-search query.

    Note this only overwrites the *working* `question` field used for the
    search call -- `original_question` is preserved so the final answer
    still directly addresses what the user actually asked.
    """
    chain = REWRITE_PROMPT | get_llm(temperature=0) | StrOutputParser()
    better_question = chain.invoke({"question": state["question"]})
    return {"question": better_question.strip()}


def web_search(state: GraphState) -> dict:
    """Node 4: query Tavily with the rewritten question and append the
    results to the working document context as a single Document."""
    tavily = TavilySearch(max_results=3, topic="general")
    response = tavily.invoke({"query": state["question"]})

    results = response.get("results", []) if isinstance(response, dict) else []
    combined_content = "\n\n".join(r.get("content", "") for r in results if r.get("content"))

    if not combined_content:
        # No usable web results -- proceed with whatever context we have
        # rather than crashing the graph.
        return {"documents": state["documents"]}

    web_doc = Document(page_content=combined_content, metadata={"source": "web_search (Tavily)"})
    return {"documents": state["documents"] + [web_doc]}


def generate(state: GraphState) -> dict:
    """Node 5: produce the final answer, strictly from the current context.

    Uses `original_question` (not the possibly-rewritten `question`) so the
    answer stays aligned with what the user actually asked.
    """
    chain = GENERATE_PROMPT | get_llm(temperature=0) | StrOutputParser()
    answer = chain.invoke(
        {"context": _format_docs(state["documents"]), "question": state["original_question"]}
    )
    return {"generation": answer}


def increment_retry(state: GraphState) -> dict:
    """Bookkeeping node: bump the retry counter before looping back to `generate`."""
    return {"generation_retries": state.get("generation_retries", 0) + 1}


def fallback(state: GraphState) -> dict:
    """Safe failure path when the generator cannot produce a grounded answer
    even after the retry budget is exhausted."""
    return {
        "generation": (
            "I wasn't able to produce an answer that is fully grounded in the "
            "retrieved 10-K excerpts or web search results for this question. "
            "Rather than risk giving you an unsupported figure, I'm flagging "
            "this rather than guessing -- please try rephrasing the question, "
            "or verify directly against the source filing."
        )
    }


def grade_generation(state: GraphState) -> str:
    """Node 6 (conditional edge): is the generation grounded in the context?"""
    if not state["documents"]:
        return "fallback"

    grader = HALLUCINATION_PROMPT | get_llm(temperature=0).with_structured_output(GradeHallucinations)
    verdict = grader.invoke(
        {"documents": _format_docs(state["documents"]), "generation": state["generation"]}
    )

    if verdict.binary_score.strip().lower() == "yes":
        return "useful"

    if state.get("generation_retries", 0) >= MAX_GENERATION_RETRIES:
        return "fallback"

    return "retry"


# ==========================================================================
# GRAPH ASSEMBLY
# ==========================================================================
def build_graph():
    """Construct and compile the CRAG state machine."""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("increment_retry", increment_retry)
    workflow.add_node("fallback", fallback)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )

    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search", "generate")

    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {"useful": END, "retry": "increment_retry", "fallback": "fallback"},
    )

    workflow.add_edge("increment_retry", "generate")
    workflow.add_edge("fallback", END)

    return workflow.compile()


def run_query(question: str) -> GraphState:
    """Convenience wrapper: run the full graph for a single question and
    return the final state (used by simple/non-streaming callers)."""
    graph = build_graph()
    initial_state: GraphState = {
        "original_question": question,
        "question": question,
        "generation": "",
        "documents": [],
        "web_search_needed": "No",
        "generation_retries": 0,
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    # Quick manual smoke test: `python agent.py "your question here"`
    import sys

    q = " ".join(sys.argv[1:]) or "What was Apple's total revenue in its most recent fiscal year?"
    result = run_query(q)
    print("\n=== ANSWER ===\n")
    print(result["generation"])
    print("\n=== SOURCES ===")
    for d in result["documents"]:
        print("-", d.metadata.get("source", "unknown"))
