from state.state import ResearchPaperState
from services.arxiv_service import ArxivService


def search_node(state: ResearchPaperState) -> ResearchPaperState:
    print("=" * 60)
    print("SEARCH NODE")
    print("=" * 60)

    papers = ArxivService.search_papers(
        query=state["user_query"],
        #max_results=1,
    )

    return {
        **state,
        "paper": papers,
    }