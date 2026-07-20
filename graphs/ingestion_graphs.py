from pathlib import Path

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from state.state import ResearchPaperState

from nodes.search_node import search_node
from nodes.download_node import download_node
from nodes.parser_node import parser_node
from nodes.chunking_node import chunking_node
from nodes.embedding_node import embedding_node


builder = StateGraph(ResearchPaperState)


# Register Nodes
# ------------------------

builder.add_node("search", search_node)
builder.add_node("download", download_node)
builder.add_node("parser", parser_node)
builder.add_node("chunking", chunking_node)
builder.add_node("embedding", embedding_node)

# ------------------------
# Build Workflow
# ------------------------

builder.add_edge(START, "search")
builder.add_edge("search", "download")
builder.add_edge("download", "parser")
builder.add_edge("parser", "chunking")
builder.add_edge("chunking", "embedding")
builder.add_edge("embedding", END)

# ------------------------
# Compile Graph
# ------------------------

ingestion_graph = builder.compile()

# ------------------------
# Save Graph Visualization
# ------------------------

output_dir = Path("graphs_output")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "ingestion_graph.png"

png_data = ingestion_graph.get_graph().draw_mermaid_png()

with open(output_file, "wb") as f:
    f.write(png_data)

print(f"Graph saved successfully to: {output_file.resolve()}")