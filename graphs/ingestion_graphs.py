from pathlib import Path

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from state.state import ResearchPaperState

from nodes.upload_node import upload_node
from nodes.parser_node import parser_node
from nodes.chunking_node import chunking_node
from nodes.embedding_node import embedding_node


builder = StateGraph(ResearchPaperState)



from langgraph.config import get_stream_writer


def with_status(stage_name, node_function):
    """Emit live processing and completion events for a graph node."""

    def wrapped_node(state):
        writer = get_stream_writer()

        writer({
            "stage": stage_name,
            "status": "processing",
        })

        result = node_function(state)

        error = result.get("error", "")

        writer({
            "stage": stage_name,
            "status": "failed" if error else "completed",
            "error": error,
        })

        return result

    return wrapped_node




# Register Nodes
# ------------------------

builder.add_node("upload", with_status("Upload", upload_node))
builder.add_node("parser", with_status("Parsing", parser_node))
builder.add_node("chunking", with_status("Chunking", chunking_node))
builder.add_node("embedding", with_status("Embeddings", embedding_node))

# ------------------------
# Build Workflow
# ------------------------

builder.add_edge(START, "upload")
builder.add_edge("upload", "parser")
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