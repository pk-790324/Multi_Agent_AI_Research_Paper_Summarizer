from typing_extensions import TypedDict,Any,List


from typing import TypedDict, Any,Optional


class ResearchPaperState(TypedDict, total=False):
    artifacts: dict[str, Any]
    error: str

from typing import TypedDict, Optional



from typing import TypedDict, Optional, Any


class ResearchState(TypedDict, total=False):

    # ========================================================
    # USER INPUT
    # ========================================================

    user_query: str

    # ========================================================
    # RETRIEVAL
    # ========================================================

    retrieved_chunks: list[dict[str, Any]]
    retrieval_count: int

    # ========================================================
    # AGENT OUTPUTS
    # ========================================================

    analysis: str
    critical_analysis: str
    cited_analysis: str
    citations: list[dict[str, Any]]

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    final_answer: str

    # ========================================================
    # ORCHESTRATOR CONTROL
    # ========================================================

    next_agent: str
    completed_agents: list[str]

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    error: Optional[str]
