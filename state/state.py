from typing_extensions import TypedDict,Any,List

class ResearchPaperState(TypedDict):
    user_query: str
    paper: list
    artifacts: dict
    chunks:list
    collection_name: str
    
    




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