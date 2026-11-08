from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import agents.analyst_agent as analyst_module
import agents.critic_agent as critic_module
import agents.orchestrator_agent as orchestrator_module


def mock_router(monkeypatch, response="critic"):
    monkeypatch.setattr(
        orchestrator_module,
        "llm",
        SimpleNamespace(
            invoke=lambda messages: SimpleNamespace(content=response)
        ),
    )


def test_rejected_critique_routes_to_analyst_and_is_saved(monkeypatch):
    critique = (
        "VERDICT: REJECTED\n"
        "REASON: The explanation is incomplete.\n"
        "REQUIRED_CORRECTIONS:\n"
        "- Explain the missing step."
    )
    mock_router(monkeypatch)

    result = orchestrator_module.orchestrator_agent({
        "retrieved_chunks": [{"page_content": "Evidence"}],
        "analysis": "Previous answer",
        "critical_analysis": critique,
    })

    assert result["next_agent"] == "analyst"
    assert result["analyst_feedback"] == critique
    assert result["critical_analysis"] == ""
    assert result["analyst_attempts"] == 1


def test_rejected_critique_stops_after_three_analyst_runs(monkeypatch):
    mock_router(monkeypatch)
    critique = "VERDICT: REJECTED\nREASON: Still incomplete."

    result = orchestrator_module.orchestrator_agent({
        "retrieved_chunks": [{"page_content": "Evidence"}],
        "analysis": "Third analyst response",
        "critical_analysis": critique,
        "analyst_attempts": 3,
    })

    assert result["next_agent"] == "citation"
    assert result["review_limit_reached"] is True
    assert result["analyst_attempts"] == 3


def test_critic_attempt_limit_stops_another_rejected_retry(monkeypatch):
    mock_router(monkeypatch)

    result = orchestrator_module.orchestrator_agent({
        "retrieved_chunks": [{"page_content": "Evidence"}],
        "analysis": "Still rejected",
        "critical_analysis": "VERDICT: REJECTED\nREASON: Incomplete.",
        "critic_attempts": 3,
        "max_critic_attempts": 3,
    })

    assert result["next_agent"] == "citation"
    assert result["review_limit_reached"] is True


def test_critic_counts_each_invocation():
    result = critic_module.critic_agent({
        "analysis": "",
        "critic_attempts": 2,
    })

    assert result["critic_attempts"] == 3


def test_more_retrieval_stops_after_three_retriever_runs(monkeypatch):
    mock_router(monkeypatch)
    critique = "VERDICT: NEED_MORE_RETRIEVAL\nREASON: More evidence needed."

    result = orchestrator_module.orchestrator_agent({
        "retrieved_chunks": [{"page_content": "Evidence"}],
        "critical_analysis": critique,
        "retrieval_attempts": 3,
    })

    assert result["next_agent"] == "citation"
    assert result["review_limit_reached"] is True
    assert result["retrieval_attempts"] == 3


def test_more_retrieval_counts_each_retriever_dispatch(monkeypatch):
    mock_router(monkeypatch)
    critique = "VERDICT: NEED_MORE_RETRIEVAL\nREASON: More evidence needed."

    result = orchestrator_module.orchestrator_agent({
        "retrieved_chunks": [{"page_content": "Evidence"}],
        "critical_analysis": critique,
        "retrieval_attempts": 2,
    })

    assert result["next_agent"] == "retriever"
    assert result["retrieval_attempts"] == 3
    assert result["retrieval_feedback"] == critique


def test_limit_handoff_proceeds_to_synthesis(monkeypatch):
    result = orchestrator_module.orchestrator_agent({
        "review_limit_reached": True,
    })

    assert result["next_agent"] == "synthesizer"


def test_analyst_receives_previous_analysis_and_critic_feedback(monkeypatch):
    previous_analysis = "The original answer."
    critique = "VERDICT: REJECTED\nREQUIRED_CORRECTIONS:\n- Add evidence."
    captured_prompt = {}

    def generate(prompt):
        captured_prompt["text"] = prompt.to_string()
        return AIMessage(content="The revised answer.")

    monkeypatch.setattr(
        analyst_module,
        "llm",
        RunnableLambda(generate),
    )

    result = analyst_module.analyst_agent({
        "user_query": "What does the evidence show?",
        "retrieved_chunks": [{"page_content": "Supporting evidence"}],
        "analysis": previous_analysis,
        "analyst_feedback": critique,
    })

    assert previous_analysis in captured_prompt["text"]
    assert critique in captured_prompt["text"]
    assert "ONLY VALID SOURCE LABELS:\n[Source 1]" in captured_prompt["text"]
    assert result["analysis"] == "The revised answer."
    assert result["analyst_feedback"] == ""
