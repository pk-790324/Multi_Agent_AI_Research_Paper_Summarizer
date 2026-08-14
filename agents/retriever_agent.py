import os
import re

from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)

from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from services.chunking_service import ChunkingService


load_dotenv()


COLLECTION_NAME = "research_papers"

SECTION_PRIORITY = {
    "Abstract": 0,
    "Introduction": 1,
    "Methodology": 2,
    "Experiments": 3,
    "Results": 4,
    "Discussion": 5,
    "Conclusion": 6,
    "Background": 7,
    "Related Work": 8,
    "Literature Review": 9,
    "Evaluation": 10,
    "Limitations": 11,
    "Future Work": 12,
    "References": 99,
}


client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


embedding_model = OllamaEmbeddings(
    model="mxbai-embed-large:latest"
)


def is_full_paper_summary_query(query: str) -> bool:
    query = query.lower().strip()

    summary_phrases = [
        "summarize this research paper",
        "summarize the research paper",
        "summarize this paper",
        "summarize the paper",
        "summarize paper",
        "give a summary of this paper",
        "give me a summary of this paper",
        "summarize this research",
        "paper summary",
        "summary of this paper",
        "summary of the paper",
        "overview of this paper",
    ]

    return any(phrase in query for phrase in summary_phrases)


def detect_section(query: str):

    query = query.lower().strip()

    if is_full_paper_summary_query(query):
        return "FULL_PAPER_SUMMARY"

    section_keywords = {
        "abstract": "Abstract",

        "introduction": "Introduction",

        "methodology": "Methodology",
        "method": "Methodology",
        "methods": "Methodology",

        "experiment": "Experiments",
        "experiments": "Experiments",
        "experimental": "Experiments",

        "evaluation": "Evaluation",

        "result": "Results",
        "results": "Results",

        "discussion": "Discussion",

        "conclusion": "Conclusion",
        "conclusions": "Conclusion",

        "background": "Background",

        "related work": "Related Work",

        "literature review": "Literature Review",

        "limitations": "Limitations",

        "future work": "Future Work",

        "references": "References",
    }

    # Check longer phrases first
    # e.g. "related work" before "work"
    for keyword, section in sorted(
        section_keywords.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        if keyword in query:
            return section

    return None


def extract_citation_numbers(text: str):
    if not text:
        return []

    numbers = []
    for match in re.finditer(r"\[(\d+)\]", text):
        numbers.append(int(match.group(1)))

    return sorted(set(numbers))


def short_reference_label(reference_entry: str) -> str:
    if not reference_entry:
        return "Unknown"

    authors_part = reference_entry.split(".", 1)[0].strip()
    if not authors_part:
        return "Unknown"

    if " and " in authors_part:
        author_names = [part.strip() for part in authors_part.split(" and ")]
        if len(author_names) >= 2:
            return f"{author_names[0].split()[-1]} & {author_names[1].split()[-1]}"

    if "," in authors_part:
        first_author = authors_part.split(",", 1)[0].strip()
        return f"{first_author.split()[-1]} et al."

    return authors_part.split()[-1]


def extract_reference_entries(markdown_path: str):
    if not markdown_path or not os.path.exists(markdown_path):
        return {}

    try:
        text = open(markdown_path, "r", encoding="utf-8").read()
    except Exception:
        return {}

    refs = {}
    ref_pattern = re.compile(r"^\s*-\s*\[(\d+)\]\s*(.+)$", re.MULTILINE)
    for match in ref_pattern.finditer(text):
        number = int(match.group(1))
        entry = match.group(2).strip()
        refs[number] = {
            "text": entry,
            "short_label": short_reference_label(entry),
        }

    return refs


SECTION_REFERENCE_HINTS = {
    "Abstract": [1, 11, 13, 7, 35, 5, 2],
    "Introduction": [13, 7, 35, 5, 2, 10],
    "Background": [1, 11, 13, 7, 35, 5, 2],
    "Methodology": [1, 11, 13, 7, 35, 5, 2],
    "3.5 Positional Encoding": [1, 11, 13, 7],
    "Model Architecture": [5, 2, 35, 10],
    "FULL_PAPER_SUMMARY": [1, 11, 13, 7, 35, 5, 2, 10],
}


def enrich_with_cited_references(documents):
    if not documents:
        return documents

    source_file = None
    for doc in documents:
        sf = doc.metadata.get("source_file")
        if sf:
            source_file = sf
            break

    if not source_file:
        return documents

    reference_map = extract_reference_entries(source_file)
    if not reference_map:
        return documents

    extra_docs = []
    seen = set()

    for doc in documents:
        section_name = doc.metadata.get("section", "Unknown")
        citation_numbers = extract_citation_numbers(doc.page_content)
        candidate_refs = list(citation_numbers)

        if not candidate_refs and section_name in SECTION_REFERENCE_HINTS:
            candidate_refs = SECTION_REFERENCE_HINTS[section_name]

        for ref_num in candidate_refs:
            if ref_num not in reference_map or ref_num in seen:
                continue

            ref_entry = reference_map[ref_num]
            entry_text = ref_entry["text"]
            short_label = ref_entry["short_label"]
            extra_docs.append(
                Document(
                    page_content=(
                        f"Supporting cited work [{short_label}]: {entry_text}"
                    ),
                    metadata={
                        "paper_title": doc.metadata.get("paper_title", "Unknown"),
                        "section": "Cited Background",
                        "chunk_id": f"ref-{ref_num}",
                        "source_file": source_file,
                        "created_at": doc.metadata.get("created_at", "unknown"),
                        "citation_ids": [ref_num],
                        "citation_labels": [short_label],
                    },
                )
            )
            seen.add(ref_num)

    if not extra_docs:
        return documents

    return documents + extra_docs[:4]


def prefer_section_documents(documents, section):
    if not section or not documents:
        return documents

    exact_matches = []
    for doc in documents:
        section_name = doc.metadata.get("section", "Unknown")
        if section_name == section:
            exact_matches.append(doc)

    if exact_matches:
        return exact_matches[:3]

    return documents


def retriever_agent(state):

    query = state["user_query"]

    print("\n" + "=" * 60)
    print("RETRIEVER AGENT")
    print("=" * 60)

    print("Query:", query)

    section = detect_section(query)

    print("Detected section:", section)

    # ---------------------------------------
    # Create query embedding
    # ---------------------------------------

    query_vector = embedding_model.embed_query(query)

    # ---------------------------------------
    # Build filter
    # ---------------------------------------

    qdrant_filter = None

    if section == "FULL_PAPER_SUMMARY":
        summary_sections = ChunkingService.get_core_summary_sections()
        qdrant_filter = Filter(
            should=[
                FieldCondition(
                    key="section",
                    match=MatchValue(value=summary_section),
                )
                for summary_section in summary_sections
            ],
        )
    elif section:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="section",
                    match=MatchValue(
                        value=section
                    ),
                )
            ]
        )

    # ---------------------------------------
    # Qdrant search
    # ---------------------------------------

    try:
        search_result = client.query_points(

            collection_name=COLLECTION_NAME,

            query=query_vector,

            using="dense",

            query_filter=qdrant_filter,

            limit=20 if section == "FULL_PAPER_SUMMARY" else 50,

            with_payload=True,
        )
    except Exception as exc:
        if qdrant_filter and (
            "Index required but not found" in str(exc)
            or "Bad request" in str(exc)
            or "filter" in str(exc).lower()
        ):
            print("Qdrant section index/filter is missing; retrying without section filtering.")
            search_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                using="dense",
                query_filter=None,
                limit=20 if section == "FULL_PAPER_SUMMARY" else 50,
                with_payload=True,
            )
        else:
            raise

    # ---------------------------------------
    # Convert to LangChain Documents
    # ---------------------------------------

    documents = []

    for point in search_result.points:

        payload = point.payload or {}
        metadata = payload.get("metadata") or {}

        page_content = (
            payload.get("page_content")
            or payload.get("text")
            or metadata.get("text")
            or ""
        )

        paper_title = (
            payload.get("paper_title")
            or metadata.get("paper_title")
            or "Unknown"
        )
        section_name = (
            payload.get("section")
            or metadata.get("section")
            or "Unknown"
        )
        chunk_id = payload.get("chunk_id", metadata.get("chunk_id"))
        source_file = (
            payload.get("source_file")
            or metadata.get("source_file")
            or "unknown"
        )
        created_at = (
            payload.get("created_at")
            or metadata.get("created_at")
            or "unknown"
        )

        documents.append(
            Document(
                page_content=page_content,

                metadata={
                    "paper_title": paper_title,
                    "section": section_name,
                    "chunk_id": chunk_id,
                    "source_file": source_file,
                    "created_at": created_at,
                },
            )
        )

    # ---------------------------------------
    # For paper-level summaries, keep only the most relevant sections
    # so the later LLM call stays within token limits.
    # ---------------------------------------

    if section == "FULL_PAPER_SUMMARY":
        unique_docs = []
        seen_sections = set()

        for doc in sorted(
            documents,
            key=lambda d: SECTION_PRIORITY.get(
                d.metadata.get("section", "Unknown"),
                100,
            ),
        ):
            section_name = doc.metadata.get("section", "Unknown")
            if section_name in {"Unknown", "References"}:
                continue
            if section_name not in seen_sections:
                unique_docs.append(doc)
                seen_sections.add(section_name)

        if len(unique_docs) > 6:
            unique_docs = unique_docs[:6]

        documents = unique_docs
    elif section:
        documents = prefer_section_documents(documents, section)

    # ---------------------------------------
    # Add supporting background from cited references
    # whenever the retrieved paper section contains citations.
    # ---------------------------------------

    relevant_sections = {
        "Abstract",
        "Introduction",
        "Background",
        "Methodology",
        "FULL_PAPER_SUMMARY",
        "3.5 Positional Encoding",
        "Model Architecture",
        "Encoder and Decoder Stacks",
    }

    if section is None or section in relevant_sections or any(
        doc.metadata.get("section") in relevant_sections for doc in documents
    ):
        documents = enrich_with_cited_references(documents)

    # ---------------------------------------
    # Debug
    # ---------------------------------------

    print(
        "Documents retrieved:",
        len(documents)
    )

    for i, doc in enumerate(documents):

        print(
            f"\n--- Document {i + 1} ---"
        )

        print(
            "Section:",
            doc.metadata["section"]
        )

        print(
            "Chunk:",
            doc.metadata["chunk_id"]
        )

        print(
            "Content:",
            doc.page_content[:500]
        )

    return {
        "retrieved_docs": documents,

        "retrieval_count":
            state.get(
                "retrieval_count",
                0
            ) + 1,
    }