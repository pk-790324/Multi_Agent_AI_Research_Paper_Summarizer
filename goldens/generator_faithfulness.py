
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from llm.llm import llm

load_dotenv()

# ----------------------------------------
# PATH CONFIGURATION
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_DIR = BASE_DIR / "chunks"

OUTPUT_FILE = (
    BASE_DIR / "goldens" / "faithfulness_goldens.json"
)

# ----------------------------------------
# LLM CONFIGURATION
# ----------------------------------------

llm=llm

# ----------------------------------------
# PROMPT
# ----------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are an expert research paper question generator.

Your task is to generate ONE meaningful question
based on the given research paper chunk.

Rules:
1. Generate a question that can be answered using
   ONLY the provided chunk.
2. The question must test understanding of the content.
3. Avoid questions about information not present in the chunk.
4. Avoid generic questions such as "What is this paper about?"
5. Return ONLY valid JSON.

Research Paper:
{paper_title}

Section:
{section}

Chunk Content:
{chunk_text}

Expected JSON format:

{{
    "query": "A meaningful question based on the chunk"
}}
""")

# ----------------------------------------
# LOAD CHUNKS
# ----------------------------------------

def load_chunks():

    all_chunks = []

    if not CHUNKS_DIR.exists():
        raise FileNotFoundError(
            f"Chunks directory not found: {CHUNKS_DIR}"
        )

    for file_path in CHUNKS_DIR.glob("*.json"):

        print(f"Loading: {file_path.name}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both a single chunk and a list of chunks
        if isinstance(data, dict):
            all_chunks.append(data)

        elif isinstance(data, list):
            all_chunks.extend(data)

    print(f"\nTotal chunks loaded: {len(all_chunks)}")

    return all_chunks


# ----------------------------------------
# GENERATE QUERY
# ----------------------------------------

def generate_query(chunk):

    chain = prompt | llm

    response = chain.invoke({
        "paper_title": chunk.get("paper_title", ""),
        "section": chunk.get("section", ""),
        "chunk_text": chunk.get("text", ""),
    })

    content = response.content.strip()

    # Handle possible Markdown JSON formatting
    content = content.replace("```json", "").replace("```", "").strip()

    result = json.loads(content)

    return result["query"]


# ----------------------------------------
# GENERATE FAITHFULNESS GOLDENS
# ----------------------------------------

def generate_faithfulness_goldens():

    chunks = load_chunks()

    goldens = []

    for index, chunk in enumerate(chunks, start=1):

        print(f"\nGenerating golden {index}/{len(chunks)}")

        chunk_text = chunk.get("text", "").strip()

        if not chunk_text:
            print("Skipping empty chunk")
            continue

        try:

            query = generate_query(chunk)

            golden = {

                # Generated question
                "query": query,

                # Original chunk as ideal context
                "ideal_context": chunk_text,

                # Original chunk metadata
                "chunk_id": chunk.get("chunk_id"),
                "paper_title": chunk.get("paper_title"),
                "section": chunk.get("section"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "source_file": chunk.get("source_file"),
                "text": chunk_text,
                "char_count": chunk.get(
                    "char_count",
                    len(chunk_text)
                ),
                "created_at": chunk.get("created_at"),

            }

            goldens.append(golden)

            print(f"Query: {query}")

        except Exception as e:

            print(f"Error processing chunk {index}: {e}")

    # ------------------------------------
    # SAVE JSON
    # ------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(
            goldens,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("FAITHFULNESS GOLDENS GENERATED")
    print("=" * 60)

    print(f"Total goldens: {len(goldens)}")
    print(f"Saved at: {OUTPUT_FILE}")


# ----------------------------------------
# ENTRY POINT
# ----------------------------------------

if __name__ == "__main__":

    generate_faithfulness_goldens()
