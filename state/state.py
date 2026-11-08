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
    retrieval_attempts: int
    retrieval_feedback: str
    analyst_attempts: int
    review_limit_reached: bool
    
    # RETRIEVAL STATUS
    retrieval_status: str
    retrieval_error: Optional[str]


    # ========================================================
    # AGENT OUTPUTS
    # ========================================================

    analysis: str
    critical_analysis: str
    analyst_feedback: str
    # In ResearchState
    critic_attempts: int
    max_critic_attempts: int
    critic_verdict: str
    cited_analysis: str
    citations: list[dict[str, Any]]
    
    
    # CITATION STATUS
    citation_status: str
    citation_error: Optional[str]

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
