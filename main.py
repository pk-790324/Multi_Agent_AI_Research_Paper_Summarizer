from graphs.ingestion_graphs import ingestion_graph
from graphs.research_graphs import research_graph
import sys


# input_state = {
#         "user_query": "https://arxiv.org/abs/2608.13558"
# }
# result=ingestion_graph.invoke(input_state)

# import sys

# result = research_graph.invoke({
#     "user_query": "explain the conclusion part of attention is all you need paper",
#     "retrieved_docs": [],
#     "analysis": "",
#     "critique": "",
#     "next_agent": "",
#     "retrieval_count": 0,
#     "final_answer": ""
# })

# print("\n" + "=" * 60)
# print("FINAL ANSWER")
# print("=" * 60)




# from graph.research_graph import research_graph


# USER_QUERY = "explain the conclusion of attention is all you need paper"

# print("\n" + "=" * 60)
# print("MAIN.PY")
# print("=" * 60)
# print("USER_QUERY =", repr(USER_QUERY))


# initial_state = {
#     "user_query": USER_QUERY,
#     "retrieved_docs": [],
#     "analysis": "",
#     "critique": "",
#     "next_agent": "",
#     "retrieval_count": 0,
#     "final_answer": "",
# }

# print("INITIAL STATE:")
# print("user_query =", repr(initial_state["user_query"]))


# result = research_graph.invoke(initial_state)

# print("\nFINAL ANSWER:")
# print(result.get("final_answer", ""))


from graphs.research_graphs import research_graph

# ============================================================
# USER QUERY
# ============================================================

USER_QUERY = "explain the experiment part omniscientists paper "


# ============================================================
# DEBUG
# ============================================================

print("\n" + "=" * 60)
print("MAIN.PY")
print("=" * 60)

print("USER_QUERY =", repr(USER_QUERY))


# ============================================================
# INITIAL STATE
# ============================================================

initial_state = {
    "user_query": USER_QUERY,
    "retrieved_docs": [],
    "analysis": "",
    "critique": "",
    "next_agent": "",
    "retrieval_count": 0,
    "final_answer": "",
}


print("\nINITIAL STATE:")
print("user_query =", repr(initial_state["user_query"]))


# ============================================================
# RUN RESEARCH GRAPH
# ============================================================

result = research_graph.invoke(
    initial_state
)


# ============================================================
# FINAL ANSWER
# ============================================================

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(
    result.get(
        "final_answer",
        ""
    )
)