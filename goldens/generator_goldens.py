
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.models import DeepEvalBaseLLM

from services.hybrid_search_service import HybridSearchService

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue
)


from judge_llm.judge_llm import judge_llm
load_dotenv()

llm = judge_llm

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

                "text": text

            })

        if next_offset is None:
            break

        offset = next_offset

    return chunks


# ==========================================
# 3. GENERATE GOLDENS FOR ONE USER
# ==========================================

def generate_goldens(current_user: dict):

    if not current_user or current_user.get("_id") is None:
        raise ValueError("An authenticated user is required.")

    owner_id = str(current_user["_id"])

    print("\n" + "=" * 50)
    print("GOLDEN DATASET GENERATION")
    print("=" * 50)

    # Initialize service inside function
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
    # 4. SAMPLE CHUNKS
    # ==========================================

    sample = random.sample(
        chunks,
        min(2, len(chunks))
    )

    contexts = [
        [chunk["text"]]
        for chunk in sample
    ]

    # ==========================================
    # 5. CUSTOM MODEL + SYNTHESIZER
    # ==========================================

    custom_llm = llm

    synthesizer = Synthesizer(
        model=custom_llm
    )

    # ==========================================
    # 6. GENERATE GOLDENS
    # ==========================================

    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
        max_goldens_per_context=1
    )

    if not goldens:
        raise ValueError(
            "DeepEval failed to generate any goldens."
        )

    # ==========================================
    # 7. CONVERT TO PROJECT SCHEMA
    # ==========================================

    if len(goldens) != len(sample):
        raise ValueError(
            f"Expected {len(sample)} goldens, "
            f"but generated {len(goldens)}. "
            "Check the generated dataset before evaluation."
        )

    rows = []

    for i, golden in enumerate(goldens):

        source_chunk = sample[i]

        rows.append({

            "id": f"g{i+1:03d}",

            "query": golden.input,

            "ideal_answer": golden.expected_output,

            "expected_retrieval_context": [
                source_chunk["text"]
            ],

            "source": {

                "owner_id": owner_id,

                "document_id": source_chunk["document_id"],

                "version": source_chunk["version"],

                "chunk_id": source_chunk["chunk_id"],

                "paper_title": source_chunk["paper_title"],

                "section": source_chunk["section"],

                "page_start": source_chunk["page_start"],

                "page_end": source_chunk["page_end"],

                "source_file": source_chunk["source_file"]

            }

        })

    # ==========================================
    # 8. SAVE OWNER-SPECIFIC DATASET
    # ==========================================

    output_dir = Path("goldens") / owner_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / "retriever_deepeval_goldens.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            rows,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 50)
    print("GOLDEN DATASET GENERATION COMPLETED")
    print("=" * 50)

    print(f"Total chunks: {len(chunks)}")
    print(f"Sampled chunks: {len(sample)}")
    print(f"Generated goldens: {len(rows)}")
    print(f"Output: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    raise SystemExit(
        "Run retriever evaluation from the authenticated frontend/API."
    )
