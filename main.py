from agents.embedding_agent import embedding_agent

response = embedding_agent.invoke(
    {
        "messages": [
            (
                "user",
                "embedd the given chunks and store in given specified locations"
            )
        ]
    }
)

print(response)