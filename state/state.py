from typing_extensions import TypedDict,Any,List


from typing import TypedDict, Any


class ResearchPaperState(TypedDict, total=False):
    artifacts: dict[str, Any]
    error: str

class ResearchState(TypedDict):

    # User input
    user_query: str

    # Retrieval
    retrieved_docs: List

    # Agent outputs
    analysis: str
    critique: str

    # Orchestration
    next_agent: str
    retrieval_count: int

    # Final response
    final_answer: str