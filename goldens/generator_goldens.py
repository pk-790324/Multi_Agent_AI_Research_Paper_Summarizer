
import os
import glob
import json
import random

from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.models import DeepEvalBaseLLM

from llm.llm import llm1
llm=llm1

load_dotenv()


# ==========================================
# 1. CUSTOM LLM
# ==========================================

class CustomLLM(DeepEvalBaseLLM):

    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "Project-LLM"


# ==========================================
# 2. LOAD EXISTING JSON CHUNKS
# ==========================================

def load_chunks():

    chunks = []

    chunk_files = glob.glob(
        "chunks/**/*.json",
        recursive=True
    )

    chunk_files += glob.glob(
        "chunks/**/*.jsonl",
        recursive=True
    )

    for path in chunk_files:

        with open(path, "r", encoding="utf-8") as f:

            if path.endswith(".jsonl"):

                data = [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]

            else:
                data = json.load(f)

                # Support a single JSON object or a list
                if isinstance(data, dict):
                    data = [data]

            for chunk in data:

                if not isinstance(chunk, dict):
                    continue

                text = chunk.get("text", "").strip()

                if not text:
                    continue

                chunks.append({

                    "text": text,

                    "chunk_id": chunk.get("chunk_id"),

                    "paper_title": chunk.get("paper_title"),

                    "section": chunk.get("section"),

                    "page_start": chunk.get("page_start"),

                    "page_end": chunk.get("page_end"),

                    "source_file": chunk.get("source_file")

                })

    return chunks


# ==========================================
# 3. LOAD AND SAMPLE
# ==========================================

chunks = load_chunks()


if not chunks:
    raise ValueError("No valid chunks found!")

print(f"Total chunks loaded: {len(chunks)}")

sample = random.sample(
    chunks,
    min(2, len(chunks))
)

# Each context contains the chunk text
contexts = [
    [chunk["text"]]
    for chunk in sample
]


# ==========================================
# 4. CUSTOM MODEL
# ==========================================

custom_llm = CustomLLM(model=llm)

synthesizer = Synthesizer(
    model=custom_llm
)


# ==========================================
# 5. GENERATE GOLDENS
# ==========================================

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=1
)



# ==========================================
# CONVERT TO PROJECT SCHEMA
# ==========================================

rows = []

for i, golden in enumerate(goldens):

    source_chunk = sample[i]

    rows.append({

        "id": f"g{i+1:03d}",

        "query": golden.input,

        "ideal_answer": golden.expected_output,

        # Original chunk used to generate this question
        "expected_retrieval_context": [
            source_chunk["text"]
        ],

        # Preserve source metadata
        "source": {

            "chunk_id": source_chunk["chunk_id"],

            "paper_title": source_chunk["paper_title"],

            "section": source_chunk["section"],

            "page_start": source_chunk["page_start"],

            "page_end": source_chunk["page_end"],

            "source_file": source_chunk["source_file"]

        }

    })

# ==========================================
# 7. SAVE DATASET
# ==========================================

os.makedirs("goldens", exist_ok=True)

output_path = (
    "goldens/retriever_deepeval_goldens.json"
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
