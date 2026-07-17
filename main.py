from agents.search_agent import search_agent

response = search_agent.invoke(
    {
        "messages": [
            (
                "user",
                "Find the paper 'Attention Is All You Need' and download its PDF.",
            )
        ]
    }
)

print(response)