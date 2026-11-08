
from services.hybrid_search_service import HybridSearchService
from evals.generator_eval_graph import generator_eval_graph
from langsmith import traceable


class RagPipeline:

    def __init__(self):
        self.retriever = HybridSearchService()

    @traceable(run_type="chain", name="RagPipeline")
    def invoke(self, query: str) -> dict:

        # 1. LIVE RETRIEVAL
        docs = self.retriever.search(query)

        if not docs:
            raise ValueError(
                f"No documents retrieved for query: {query}"
            )

        # 2. EXTRACT RETRIEVAL CONTEXT
        context = []

        for doc in docs:
            content = (
                doc.get("page_content")
                or doc.get("text")
                or doc.get("content")
            )

            if not content:
                raise ValueError(
                    "Retrieved document has no recognizable text field."
                )

            context.append(content)

        # 3. PREPARE GENERATOR STATE
        initial_state = {
            "user_query": query,
            "retrieved_chunks": docs,
            "retrieval_count": len(docs),
            "analysis": "",
            "critical_analysis": "",
            "cited_analysis": "",
            "citations": [],
            "final_answer": "",
            "next_agent": "retriever",
            "completed_agents": [],
            "error": None,
        }

        # 4. MULTI-AGENT GENERATION
        result = generator_eval_graph.invoke(
            initial_state,
            config={"recursion_limit": 30}
        )

        if result.get("error"):
            raise RuntimeError(result["error"])

        answer = result.get("final_answer", "")

        if not answer:
            raise ValueError("Generator returned an empty final answer.")

        # 5. RETURN EVALUATION DATA
        return {
            "query": query,
            "context": context,
            "retrieved_documents": docs,
            "answer": answer,
            "agent_outputs": result,
        }


if __name__ == "__main__":

    rag = RagPipeline()

    result = rag.invoke(
        "Summarize the era of 1-bit LLM papers in short."
    )

    print("\nQUERY:", result["query"])
    print("\nANSWER:", result["answer"])

    print("\nCONTEXT CHUNKS:")

    for i, chunk in enumerate(result["context"]):
        print(f"\n[{i}] {chunk[:120]}...")
