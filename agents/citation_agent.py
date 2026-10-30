
import re

from langchain_core.prompts import ChatPromptTemplate

from state.state import ResearchState
from llm.llm import llm


# ============================================================
# CITATION PROMPT
# ============================================================

citation_prompt = ChatPromptTemplate.from_template("""
You are an academic citation agent.

Your task is to connect factual claims in the analyst's
response to the original retrieved research-paper sources.

USER QUESTION:
{user_query}

ANALYST'S RESPONSE:
{analysis}

AVAILABLE RESEARCH-PAPER SOURCES:
{sources}

INSTRUCTIONS:

1. Read the analyst's response carefully.
2. Identify its important factual claims.
3. Match each claim with the source that directly supports it.
4. Add citations using ONLY the supplied source IDs.
5. Use citation format [S1], [S2], etc.
6. A claim may have multiple citations if multiple sources support it.
7. Never invent paper titles, sections, chunk IDs, or page numbers.
8. Do not cite a source unless its content supports the claim.
9. If a claim has no supporting source, mark it [Unverified].
10. Preserve the technical meaning of the analyst's response.
11. Do not create new source IDs.
12. Do not treat a source title alone as evidence; examine its content.

OUTPUT FORMAT:

### Answer with Citations

Provide the analyst's answer with citations attached
to the relevant claims.

### References

List each source used in this format:

[S1] Paper Title - Section - Page - Chunk ID

### Unsupported Claims

Mention claims that could not be verified using
the retrieved sources.

IMPORTANT:
Only cite sources from the provided source list.
""")


# ============================================================
# CITATION AGENT
# ============================================================

def citation_agent(state: ResearchState):

    print("\n" + "=" * 60)
    print("CITATION AGENT")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. READ STATE
    # --------------------------------------------------------

    query = state.get("user_query", "").strip()

    analysis = state.get("analysis", "").strip()

    chunks = state.get(
        "retrieved_chunks",
        []
    )

    print("Research question:", query)
    print("Retrieved chunks:", len(chunks))
    print("Analysis available:", bool(analysis))

    # --------------------------------------------------------
    # 2. VALIDATE INPUT
    # --------------------------------------------------------

    if not analysis:

        return {
            "cited_analysis": "",
            "citations": [],
            "citation_status": "failed",
            "citation_error": "Analyst response is empty."
        }

    if not chunks:

        return {
            "cited_analysis": analysis,
            "citations": [],
            "citation_status": "no_sources",
            "citation_error": "No retrieved sources available."
        }

    # --------------------------------------------------------
    # 3. PREPARE SOURCE INFORMATION
    # --------------------------------------------------------

    MAX_SOURCES = 8
    MAX_SOURCE_CHARS = 2500

    source_catalog = []
    source_map = {}

    for index, chunk in enumerate(
        chunks[:MAX_SOURCES],
        start=1
    ):

        source_id = f"S{index}"

        # Support both possible text field names
        content = chunk.get(
            "page_content",
            chunk.get("text", "")
        )

        content = content.strip() if content else ""

        if not content:
            continue

        source = {
            "source_id": source_id,
            "paper_title": chunk.get(
                "paper_title",
                "Unknown"
            ),
            "section": chunk.get(
                "section",
                "Unknown"
            ),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "chunk_id": chunk.get(
                "chunk_id",
                "Unknown"
            ),
            "source_file": chunk.get(
                "source_file",
                "Unknown"
            ),
            "content": content[:MAX_SOURCE_CHARS]
        }

        source_map[source_id] = source
        source_catalog.append(source)

    if not source_catalog:

        return {
            "cited_analysis": analysis,
            "citations": [],
            "citation_status": "no_sources",
            "citation_error": "Retrieved chunks contain no usable text."
        }

    # --------------------------------------------------------
    # 4. FORMAT SOURCE CONTEXT
    # --------------------------------------------------------

    source_parts = []

    for source in source_catalog:

        page_start = source["page_start"]
        page_end = source["page_end"]

        if page_start is not None and page_end is not None:

            page_info = (
                str(page_start)
                if page_start == page_end
                else f"{page_start}-{page_end}"
            )

        else:
            page_info = "Unknown"

        source_text = f"""
Source ID: [{source['source_id']}]
Paper: {source['paper_title']}
Section: {source['section']}
Page: {page_info}
Chunk ID: {source['chunk_id']}

Content:
{source['content']}
"""

        source_parts.append(source_text)

    sources = "\n\n".join(source_parts)

    print("Available sources:", len(source_catalog))

    # --------------------------------------------------------
    # 5. GENERATE CITATIONS
    # --------------------------------------------------------

    try:

        chain = citation_prompt | llm

        response = chain.invoke({
            "user_query": query,
            "analysis": analysis,
            "sources": sources
        })

        cited_analysis = response.content.strip()

    except Exception as e:

        print("Citation error:", str(e))

        return {
            "cited_analysis": analysis,
            "citations": [],
            "citation_status": "failed",
            "citation_error": str(e)
        }

    # --------------------------------------------------------
    # 6. VALIDATE CITATION IDS
    # --------------------------------------------------------

    allowed_ids = set(source_map.keys())

    # Find all citation IDs generated by the LLM
    used_ids = set(
        re.findall(r"\[(S\d+)\]", cited_analysis)
    )

    invalid_ids = used_ids - allowed_ids

    # Remove invalid source references
    for invalid_id in invalid_ids:

        cited_analysis = cited_analysis.replace(
            f"[{invalid_id}]",
            "[Invalid citation removed]"
        )

    # Only retain citations that exist in the source map
    valid_used_ids = used_ids & allowed_ids

    citations = []

    for source_id in sorted(
        valid_used_ids,
        key=lambda value: int(value[1:])
    ):

        source = source_map[source_id]

        citations.append({
            "source_id": source_id,
            "paper_title": source["paper_title"],
            "section": source["section"],
            "page_start": source["page_start"],
            "page_end": source["page_end"],
            "chunk_id": source["chunk_id"],
            "source_file": source["source_file"]
        })

    # --------------------------------------------------------
    # 7. RETURN RESULTS
    # --------------------------------------------------------

    print("Valid citations used:", len(citations))
    print("Invalid citations detected:", len(invalid_ids))

    return {
        "cited_analysis": cited_analysis,
        "citations": citations,
        "citation_status": "success",
        "citation_error": None
    }
