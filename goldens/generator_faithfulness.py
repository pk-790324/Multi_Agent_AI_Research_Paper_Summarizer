
import json
import random
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from services.hybrid_search_service import HybridSearchService
from llm.llm import gemma

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue
)

load_dotenv()

generator_llm =gemma


# ==========================================
# 1. PROMPT
# ==========================================

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
6. Do not include Markdown code fences.

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


# ==========================================
# 2. LOAD USER CHUNKS FROM QDRANT
# ==========================================

def load_chunks_from_qdrant(
    client,
    owner_id: str,
    collection_name="research_papers"
):

    if not owner_id:
        raise ValueError("owner_id is required")

    chunks = []

    offset = None

    while True:

        points, next_offset = client.scroll(

            collection_name=collection_name,

            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="owner_id",
                        match=MatchValue(value=owner_id)
                    )
                ]
            ),

            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for point in points:

            payload = point.payload or {}

            text = (
                payload.get("page_content") or ""
            ).strip()

            if not text:
                continue

            chunks.append({

                "chunk_id": payload.get("chunk_id"),

                "owner_id": payload.get("owner_id"),

                "document_id": payload.get("document_id"),

                "version": payload.get("version"),

                "paper_title": payload.get("paper_title"),

                "section": payload.get("section"),

                "page_start": payload.get("page_start"),

                "page_end": payload.get("page_end"),

                "source_file": payload.get("source_file"),

                "created_at": payload.get("created_at"),

                "char_count": len(text),

                "text": text

            })

        if next_offset is None:
            break

        offset = next_offset

    return chunks


# ==========================================
# 3. GENERATE QUERY
# ==========================================

def generate_query(chunk):

    chain = prompt | generator_llm

    response = chain.invoke({

        "paper_title": chunk.get("paper_title") or "Unknown",

        "section": chunk.get("section") or "Unknown",

        "chunk_text": chunk.get("text", "")

    })

    content = response.content.strip()

    # Remove Markdown code fences
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE
    ).strip()

    # Parse JSON, allowing surrounding explanatory text
    start = content.find("{")

    if start == -1:
        raise ValueError("LLM did not return a JSON object.")

    result, _ = json.JSONDecoder().raw_decode(
        content[start:]
    )

    query = result.get("query")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Generated query is missing or invalid.")

    return query.strip()


# ==========================================
# 4. GENERATE GOLDENS FOR ONE USER
# ==========================================

def generate_faithfulness_goldens(current_user: dict):

    if not current_user or current_user.get("_id") is None:
        raise ValueError("An authenticated user is required.")

    owner_id = str(current_user["_id"])

    print("\n" + "=" * 50)
    print("GENERATOR GOLDEN DATASET GENERATION")
    print("=" * 50)

    # Initialize Qdrant service
    hybrid_service = HybridSearchService()

    # Load only this user's chunks
    chunks = load_chunks_from_qdrant(
        client=hybrid_service.client,
        owner_id=owner_id
    )

    if not chunks:
        raise ValueError(
            "No chunks found for this user!"
        )

    print(f"Total chunks loaded: {len(chunks)}")

    # ==========================================
    # 5. SAMPLE CHUNKS
    # ==========================================

    sample = random.sample(
        chunks,
        min(2, len(chunks))
    )

    print(f"Sampled chunks: {len(sample)}")

    # ==========================================
    # 6. GENERATE GOLDENS
    # ==========================================

    goldens = []

    for index, chunk in enumerate(sample, start=1):

        print(
            f"\nGenerating golden {index}/{len(sample)}"
        )

        try:

            query = generate_query(chunk)

            chunk_text = chunk["text"]

            golden = {

                "id": f"g{index:03d}",

                "owner_id": owner_id,

                # Generated question
                "query": query,

                # Original chunk as ideal context
                "ideal_context": chunk_text,

                # Original chunk metadata
                "chunk_id": chunk.get("chunk_id"),

                "document_id": chunk.get("document_id"),

                "version": chunk.get("version"),

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

                "created_at": chunk.get("created_at")

            }

            goldens.append(golden)

            print(f"Generated query: {query}")

        except Exception as e:

            print(
                f"Error processing chunk {index}: {e}"
            )

    if not goldens:
        raise ValueError(
            "Failed to generate any faithfulness goldens."
        )

    # ==========================================
    # 7. SAVE OWNER-SPECIFIC DATASET
    # ==========================================

    output_dir = Path("goldens") / owner_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / "faithfulness_goldens.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            goldens,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ==========================================
    # 8. SUMMARY
    # ==========================================

    print("\n" + "=" * 50)
    print("GENERATOR GOLDEN DATASET COMPLETED")
    print("=" * 50)

    print(f"Owner ID: {owner_id}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Sampled chunks: {len(sample)}")
    print(f"Generated goldens: {len(goldens)}")
    print(f"Output: {output_path}")

    return str(output_path)


# ==========================================
# 9. ENTRY POINT
# ==========================================

if __name__ == "__main__":

    raise SystemExit(
        "Run generator golden generation "
        "from the authenticated frontend/API."
    )
