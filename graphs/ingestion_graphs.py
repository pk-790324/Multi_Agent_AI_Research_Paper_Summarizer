from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.graph import StateGraph
from state.state import ResearchPaperState

from agents.search_agent import search_agent
from agents.parser_agent import parser_agent
from agents.chunking_agents import chunking_agent
from agents.embedding_agent import embedding_agent


builder = StateGraph(ResearchPaperState)

builder.add_node("search",search_agent)
builder.add_node("parser",parser_agent)
builder.add_node("chunking",chunking_agent)
builder.add_node("embedding",embedding_agent)


builder.add_edge(START, "search")
builder.add_edge("search", "parser")
builder.add_edge("parser", "chunking")
builder.add_edge("chunking", "embedding")
builder.add_edge("embedding", END)

ingestion_graph = builder.compile()

from pathlib import Path

# Create the output directory if it doesn't exist
output_dir = Path("graphs_output")
output_dir.mkdir(exist_ok=True)

# Define the output file
output_file = output_dir / "ingestion_graph.png"

# Generate the graph image and save it
png_data = ingestion_graph.get_graph().draw_mermaid_png()

with open(output_file, "wb") as f:
    f.write(png_data)

print(f"Graph saved successfully to: {output_file.resolve()}")