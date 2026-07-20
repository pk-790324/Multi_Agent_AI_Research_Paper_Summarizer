from graphs.ingestion_graphs import ingestion_graph

input_state = {
        "user_query": "https://arxiv.org/abs/2607.16112"
}
result=ingestion_graph.invoke(input_state)