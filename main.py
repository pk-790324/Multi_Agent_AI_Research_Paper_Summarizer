from graphs.ingestion_graphs import ingestion_graph
from graphs.research_graphs import research_graph


# input_state = {
#         "user_query": "https://arxiv.org/abs/1706.03762"
# }
# result=ingestion_graph.invoke(input_state)


result = research_graph.invoke({
    "user_query": "explain the optimizer part of the paper",
    "retrieved_docs": [],
    "analysis": "",
    "critique": "",
    "next_agent": "",
    "retrieval_count": 0,
    "final_answer": ""
})

print(result["final_answer"])

