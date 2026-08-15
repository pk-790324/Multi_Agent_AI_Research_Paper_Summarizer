import os
import re
import sys

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


def normalize_paper_slug(title: str) -> str:
    """Normalize the full paper title to a safe slug for metadata filtering.

    IMPORTANT: Do NOT strip the subtitle (after ':') here — the embedding
    service slugifies the *full* title, so we must do the same to get a match.
    """
    if not title:
        return ""
    # Remove only parenthetical suffixes, keep subtitles intact
    base_title = title.split("(", 1)[0].strip()
    return re.sub(r"[^a-z0-9]+", "-", base_title.lower()).strip("-")


def infer_target_title(query: str):
    """Infer the *full stored* paper title from natural-language queries.

    The returned title must match exactly what was indexed so that
    normalize_paper_slug() produces the correct slug for filtering.
    """
    q = (query or "").lower().strip()
    if not q:
        return None

    q = re.sub(r"[^a-z0-9\s-]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    # Map user-facing aliases to the EXACT paper_title stored in Qdrant
    aliases = {
        "omniscientist": "OmniScientist: An Omni-Modal Omni-Discipline AI Scientist",
        "omni scientist": "OmniScientist: An Omni-Modal Omni-Discipline AI Scientist",
        "omni-scientist": "OmniScientist: An Omni-Modal Omni-Discipline AI Scientist",
        "omniscientists": "OmniScientist: An Omni-Modal Omni-Discipline AI Scientist",
        "attention is all you need": "Attention Is All You Need",
        "transformer": "Attention Is All You Need",
    }

    for alias, title in aliases.items():
        if alias in q:
            return title

    return None


def build_qdrant_filter(query: str, section: str | None = None):
    """Build a Qdrant filter that narrows retrieval to the target paper plus an optional section."""
    target_title = infer_target_title(query)
    paper_conditions = []

    if target_title:
        paper_slug = normalize_paper_slug(target_title)
        paper_conditions.append(
            FieldCondition(
                key="paper_slug",
                match=MatchValue(value=paper_slug),
            )
        )

    if section == "FULL_PAPER_SUMMARY":
        summary_sections = ChunkingService.get_core_summary_sections()
        section_conditions = [
            FieldCondition(
                key="section",
                match=MatchValue(value=summary_section),
            )
            for summary_section in summary_sections
        ]
        if paper_conditions:
            return Filter(must=paper_conditions, should=section_conditions)
        return Filter(should=section_conditions)

    if section:
        section_condition = FieldCondition(
            key="section",
            match=MatchValue(value=section),
        )
        if paper_conditions:
            return Filter(must=paper_conditions + [section_condition])
        return Filter(must=[section_condition])

    if paper_conditions:
        return Filter(must=paper_conditions)

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

    # ---------------------------------------
    # Infinite-loop guard
    # If we have already retried 3 times with no results,
    # return an empty result so the orchestrator can escape.
    # ---------------------------------------

    retrieval_count = state.get("retrieval_count", 0)
    if retrieval_count >= 3:
        print("[RETRIEVER] Reached max retries (3). Returning empty docs to allow synthesizer fallback.")
        return {
            "retrieved_docs": [],
            "retrieval_count": retrieval_count + 1,
        }

    section = detect_section(query)

    print("Detected section:", section)

    # ---------------------------------------
    # Create query embedding
    # ---------------------------------------

    query_vector = embedding_model.embed_query(query)

    # ---------------------------------------
    # Build filters for graduated fallback:
    #   1. paper_slug + section  (most specific)
    #   2. paper_slug only       (drop section)
    #   3. no filter             (pure semantic)
    # This avoids mixing documents from unrelated papers
    # when the target paper simply uses different section names.
    # ---------------------------------------

    full_filter      = build_qdrant_filter(query, section)          # slug + section
    paper_only_filter = build_qdrant_filter(query, None)            # slug only
    limit = 20 if section == "FULL_PAPER_SUMMARY" else 50

    def _query(q_filter):
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            using="dense",
            query_filter=q_filter,
            limit=limit,
            with_payload=True,
        )

    def _safe_search():
        """Try the full filter, fall back progressively."""
        # Step 1: full filter (paper + section)
        try:
            result = _query(full_filter)
            if result.points:
                return result
        except Exception as exc:
            if "Index" in str(exc) or "Bad request" in str(exc) or "filter" in str(exc).lower():
                print("[RETRIEVER] Filter error on full filter:", exc)
            else:
                raise

        # Step 2: paper-slug only (drop section)
        if paper_only_filter and paper_only_filter != full_filter:
            print("[RETRIEVER] Full filter returned 0 results. "
                  "Retrying with paper-slug only (no section filter).")
            try:
                result = _query(paper_only_filter)
                if result.points:
                    return result
            except Exception:
                pass

        # Step 3: no filter (pure semantic search)
        if full_filter is not None or paper_only_filter is not None:
            print("[RETRIEVER] Paper filter returned 0 results. "
                  "Retrying without any filter (semantic-only).")
        return _query(None)

    search_result = _safe_search()

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
    # Debug (unicode-safe for Windows console)
    # ---------------------------------------

    def _safe_print(text: str) -> None:
        """Write text to stdout safely, replacing un-encodable characters.

        On Windows the default stdout encoding (cp1252) cannot represent many
        Unicode codepoints found in academic papers (math symbols, etc.).
        We encode to the stdout codec with 'replace' error handling and write
        the resulting bytes directly to sys.stdout.buffer so the text always
        reaches the console without raising UnicodeEncodeError.
        """
        enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        safe_bytes = (text + "\n").encode(enc, errors="replace")
        sys.stdout.buffer.write(safe_bytes)
        sys.stdout.buffer.flush()

    _safe_print(f"Documents retrieved: {len(documents)}")

    for i, doc in enumerate(documents):
        _safe_print(f"\n--- Document {i + 1} ---")
        _safe_print(f"Section: {doc.metadata.get('section', 'Unknown')}")
        _safe_print(f"Chunk: {doc.metadata.get('chunk_id', 'Unknown')}")
        snippet = doc.page_content[:500]
        _safe_print(f"Content: {snippet}")

    return {
        "retrieved_docs": documents,

        "retrieval_count":
            state.get(
                "retrieval_count",
                0
            ) + 1,
    }









