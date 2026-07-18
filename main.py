from agents.retrievers_agent import retriever_agent

response = retriever_agent.invoke(
    {
        "messages": [
            (
                "user",
                "retrieve the methodology topic"
            )
        ]
    }
)

print(response)