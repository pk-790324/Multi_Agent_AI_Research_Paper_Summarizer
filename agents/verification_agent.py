import json
from typing import Dict, List

from langchain_ollama import ChatOllama
from langchain_core.documents import Document


class VerificationAgent:
    def __init__(self):
        """
        Initialize the verification agent with the Ollama model.
        """
        print("Initializing VerificationAgent with Ollama...")

        self.model = ChatOllama(
            model="minimax-m3:cloud",
            temperature=0.0
        )

        print("Ollama model initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, answer: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to verify the answer against the context.
        """
        prompt = f"""
You are an AI assistant designed to verify the accuracy and relevance of answers based on the provided context.

Instructions:
- Verify the following answer against the provided context.
- Check for:
1. Direct/indirect factual support (YES/NO)
2. Unsupported claims (list any if present)
3. Contradictions (list any if present)
4. Relevance to the question (YES/NO)
- Provide additional details or explanations where relevant.
- Respond in the exact format specified below without adding any unrelated information.

Format:
Supported: YES/NO
Unsupported Claims: [item1, item2, ...]
Contradictions: [item1, item2, ...]
Relevant: YES/NO
Additional Details: [Any extra information or explanations]

Answer:
{answer}

Context:
{context}

Respond ONLY with the above format.
"""
        return prompt

    def parse_verification_response(self, response_text: str) -> Dict:
        """
        Parse the LLM's verification response into a structured dictionary.
        """
        try:
            lines = response_text.split('\n')
            verification = {}

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().title()
                    value = value.strip()

                    if key in {
                        "Supported",
                        "Unsupported Claims",
                        "Contradictions",
                        "Relevant",
                        "Additional Details"
                    }:

                        if key in {"Unsupported Claims", "Contradictions"}:
                            if value.startswith('[') and value.endswith(']'):
                                items = value[1:-1].split(',')
                                items = [
                                    item.strip().strip('"').strip("'")
                                    for item in items if item.strip()
                                ]
                                verification[key] = items
                            else:
                                verification[key] = []

                        elif key == "Additional Details":
                            verification[key] = value

                        else:
                            verification[key] = value.upper()

            # Ensure all keys exist
            defaults = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": ""
            }

            for key, value in defaults.items():
                verification.setdefault(key, value)

            return verification

        except Exception as e:
            print(f"Error parsing verification response: {e}")
            return None

    def format_verification_report(self, verification: Dict) -> str:
        """
        Format the verification report dictionary into a readable paragraph.
        """
        report = f"Supported: {verification.get('Supported', 'NO')}\n"

        unsupported_claims = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])

        report += (
            f"Unsupported Claims: {', '.join(unsupported_claims)}\n"
            if unsupported_claims
            else "Unsupported Claims: None\n"
        )

        report += (
            f"Contradictions: {', '.join(contradictions)}\n"
            if contradictions
            else "Contradictions: None\n"
        )

        report += f"Relevant: {verification.get('Relevant', 'NO')}\n"

        additional_details = verification.get("Additional Details", "")

        report += (
            f"Additional Details: {additional_details}\n"
            if additional_details
            else "Additional Details: None\n"
        )

        return report

    def check(self, answer: str, documents: List[Document]) -> Dict:
        """
        Verify the answer against the provided documents.
        """
        print(
            f"VerificationAgent.check called with answer='{answer}' and {len(documents)} documents."
        )

        context = "\n\n".join(
            doc.page_content for doc in documents
        )

        print(f"Combined context length: {len(context)} characters.")

        prompt = self.generate_prompt(answer, context)

        try:
            print("Sending prompt to the model...")

            response = self.model.invoke(prompt)

            print("LLM response received.")

            llm_response = response.content.strip()

            print(f"Raw LLM response:\n{llm_response}")

        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError(
                "Failed to verify answer due to a model error."
            ) from e

        sanitized_response = (
            self.sanitize_response(llm_response)
            if llm_response
            else ""
        )

        if not sanitized_response:
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Empty response from the model."
            }

        else:
            verification_report = self.parse_verification_response(
                sanitized_response
            )

            if verification_report is None:
                verification_report = {
                    "Supported": "NO",
                    "Unsupported Claims": [],
                    "Contradictions": [],
                    "Relevant": "NO",
                    "Additional Details": "Failed to parse the model's response."
                }

        verification_report_formatted = self.format_verification_report(
            verification_report
        )

        print(f"Verification report:\n{verification_report_formatted}")

        return {
            "verification_report": verification_report_formatted,
            "context_used": context
        }