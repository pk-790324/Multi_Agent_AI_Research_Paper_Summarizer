from langchain_ollama import ChatOllama
import logging

logger = logging.getLogger(__name__)


class RelevanceChecker:
    def __init__(self):
        # Initialize the Ollama model
        self.model = ChatOllama(
            model="minimax-m3:cloud",
            temperature=0
        )

    def check(self, question: str, retriever, k=3) -> str:
        """
        1. Retrieve the top-k document chunks from the retriever.
        2. Combine them into a single text string.
        3. Pass that text + question to the LLM for classification.

        Returns:
            "CAN_ANSWER", "PARTIAL", or "NO_MATCH"
        """

        logger.debug(
            f"RelevanceChecker.check called with question='{question}' and k={k}"
        )

        # Retrieve document chunks
        top_docs = retriever.invoke(question)

        if not top_docs:
            logger.debug(
                "No documents returned from retriever.invoke(). Classifying as NO_MATCH."
            )
            return "NO_MATCH"

        # Combine the top k chunks
        document_content = "\n\n".join(
            doc.page_content for doc in top_docs[:k]
        )

        # Create prompt
        prompt = f"""
You are an AI relevance checker between a user's question and provided document content.

Instructions:
- Classify how well the document content addresses the user's question.
- Respond with only one of the following labels:
  CAN_ANSWER
  PARTIAL
  NO_MATCH
- Do not include any explanation.

Labels:
1. CAN_ANSWER:
The passages contain enough explicit information to fully answer the question.

2. PARTIAL:
The passages mention or discuss the topic but do not provide all details needed for a complete answer.

3. NO_MATCH:
The passages do not discuss or mention the question's topic at all.

Important:
If the passages mention the topic in any way, even if incomplete, respond with PARTIAL instead of NO_MATCH.

Question:
{question}

Passages:
{document_content}

Respond ONLY with one of:
CAN_ANSWER, PARTIAL, NO_MATCH
"""

        try:
            response = self.model.invoke(prompt)
            llm_response = response.content.strip().upper()

            logger.debug(f"LLM response: {llm_response}")

        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            return "NO_MATCH"

        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}

        # Default classification
        classification = "NO_MATCH"

        for label in valid_labels:
            if label in llm_response:
                logger.debug(f"Classification recognized as '{label}'.")
                classification = label
                break

        print(f"Checker response: {classification}")

        return classification