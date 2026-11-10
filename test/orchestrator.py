
from agents.orchestrator_agent import orchestrator_agent

# Create a sample state
state = {
    "user_query": "Explain the attention mechanism",
    "retrieved_chunks": [
        {"page_content": "Attention calculates relationships between tokens."}
    ],
    "analysis": "",
    "critical_analysis": "",
    "cited_analysis": "",
    "retrieval_attempts": 0,
    "analyst_attempts": 0,
    "critic_attempts": 0,
    "max_critic_attempts": 3,
    "review_limit_reached": False,
}

# Execute Orchestrator
result = orchestrator_agent(state)

# Print complete result
print("\n========== ORCHESTRATOR RESULT ==========")

print("LLM Selected Agent:", result.get("llm_selected_agent"))
print("Final Next Agent:", result.get("next_agent"))
print("Routing Reason:", result.get("routing_reason"))

# Verify returned fields
assert "llm_selected_agent" in result
assert "next_agent" in result

print("\nTEST PASSED!")
