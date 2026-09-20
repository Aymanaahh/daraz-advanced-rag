"""
Groq LLM generator for the Daraz Advanced RAG application.

Generates grounded answers using retrieved evidence and requires
source citations in the format:

[SOURCE: CHUNK-ID]
"""

import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class GroqGenerator:
    """Generate grounded answers using a Groq-hosted LLM."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ):
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. "
                "Add GROQ_API_KEY to Streamlit Cloud Secrets."
            )

        self.model = model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        )

        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = Groq(api_key=self.api_key)

    @staticmethod
    def _get_value(item: Any, key: str, default: Any = None) -> Any:
        """Safely retrieve a value from dictionaries or objects."""
        if isinstance(item, dict):
            return item.get(key, default)

        return getattr(item, key, default)

    def _format_evidence(self, evidence: List[Any]) -> str:
        """Convert retrieved evidence into a grounded context block."""

        formatted = []

        for i, item in enumerate(evidence, start=1):
            chunk_id = (
                self._get_value(item, "chunk_id")
                or self._get_value(item, "id")
                or self._get_value(item, "citation_id")
                or f"EVIDENCE-{i}"
            )

            text = (
                self._get_value(item, "text")
                or self._get_value(item, "content")
                or self._get_value(item, "chunk_text")
                or ""
            )

            domain = self._get_value(item, "domain", "")
            document_type = self._get_value(
                item,
                "document_type",
                "",
            )

            formatted.append(
                f"""SOURCE ID: {chunk_id}
DOMAIN: {domain}
DOCUMENT TYPE: {document_type}
CONTENT:
{text}
"""
            )

        return "\n-------------------------\n".join(formatted)

    def _build_prompt(
        self,
        query: str,
        evidence: List[Any],
    ) -> str:
        """Build a grounded RAG prompt."""

        context = self._format_evidence(evidence)

        return f"""
You are a professional Daraz Operations and Policy Decision Support Assistant.

Answer the user's question using ONLY the retrieved evidence provided below.

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{context}

STRICT RULES:

1. Use only information supported by the retrieved evidence.
2. Do not invent policies, procedures, deadlines, fees, eligibility rules,
   or operational facts.
3. If the evidence does not contain enough information, clearly say that
   the available knowledge base does not contain enough information.
4. Every factual claim based on retrieved evidence must have a citation.
5. Use citations exactly in this format:

[SOURCE: CHUNK-ID]

6. Use the actual SOURCE ID supplied in the evidence.
7. Do not create fake source IDs.
8. When multiple sources support a statement, cite each relevant source.
9. Keep the answer concise but useful.
10. Prefer bullet points when explaining procedures, timelines, or conditions.
11. Do not mention internal RAG implementation details unless the user asks.

ANSWER:
""".strip()

    def generate(
        self,
        query: str,
        evidence: List[Any],
    ) -> str:
        """
        Generate a grounded answer.

        Parameters
        ----------
        query:
            User's question.

        evidence:
            Retrieved evidence chunks.

        Returns
        -------
        str
            LLM-generated grounded answer.
        """

        if not query or not query.strip():
            return "Please provide a question."

        if not evidence:
            return (
                "I could not find relevant information in the available "
                "knowledge base."
            )

        prompt = self._build_prompt(query, evidence)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a factual enterprise operations assistant. "
                        "Ground every answer in the supplied evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content

        if not answer:
            return (
                "The language model returned an empty response. "
                "Please try the question again."
            )

        return answer.strip()

    def get_model_info(self) -> Dict[str, Any]:
        """Return generator configuration for the Streamlit UI."""

        return {
            "provider": "Groq",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def extract_citations(answer: str) -> List[str]:
    """
    Extract [SOURCE: CHUNK-ID] citations from an answer.
    """

    if not answer:
        return []

    pattern = r"\[SOURCE:\s*([^\]]+)\]"

    citations = re.findall(pattern, answer)

    # Preserve order while removing duplicates.
    unique = []

    for citation in citations:
        citation = citation.strip()

        if citation and citation not in unique:
            unique.append(citation)

    return unique


if __name__ == "__main__":
    """
    Basic standalone smoke test.

    This requires GROQ_API_KEY to be configured.
    """

    generator = GroqGenerator()

    test_evidence = [
        {
            "chunk_id": "REF-002-P01-C001",
            "domain": "refunds",
            "document_type": "refund_timelines",
            "text": (
                "Refund processing timelines depend on the payment method "
                "used for the original transaction."
            ),
        },
        {
            "chunk_id": "REF-002-P01-C002",
            "domain": "refunds",
            "document_type": "refund_timelines",
            "text": (
                "Bank-related refunds may require additional processing "
                "time after the refund has been initiated."
            ),
        },
    ]

    question = "How long does a refund take?"

    answer = generator.generate(
        question,
        test_evidence,
    )

    print("\nGenerated answer:\n")
    print(answer)

    print("\nExtracted citations:")
    print(extract_citations(answer))