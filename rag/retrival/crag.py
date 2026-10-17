"""
Corrective RAG (CRAG) Implementation for Medical Guidelines.
Reference: Corrective Retrieval Augmented Generation (Yan et al.)

Key Components:
1. Document Relevance Evaluator:
   - Grades each retrieved document chunk as CORRECT, AMBIGUOUS, or INCORRECT.
   - Assigns a numerical relevance confidence score.
2. Knowledge Refinement:
   - Strips non-salient noise and extracts focused 'knowledge strips' (clinically pertinent excerpts).
3. Corrective Fallback & Routing:
   - If retrieval confidence is low (all chunks INCORRECT), triggers query reformulation and secondary search.
   - If secondary search still yields no matching internal guidelines, safely routes to a verified disclaimer fallback instead of hallucinating.
"""

from typing import List, Dict, Any, Tuple
import json
from pydantic import BaseModel, Field
from openai import OpenAI
from langchain_core.documents import Document


class GradedDocument(BaseModel):
    document: Dict[str, Any]
    grade: str  # "CORRECT", "AMBIGUOUS", "INCORRECT"
    relevance_score: float  # 0.0 to 1.0
    rationale: str
    refined_strip: str = ""  # Extracted salient clinical knowledge strip


class CRAGDecision(BaseModel):
    overall_status: str  # "CORRECT", "AMBIGUOUS", "INCORRECT"
    confidence_score: float
    graded_documents: List[GradedDocument] = Field(default_factory=list)
    refined_context: str = ""
    trigger_fallback: bool = False
    fallback_query: str = ""
    feedback: str = ""


class CorrectiveRAG:
    """
    Implements CRAG evaluation, filtering, knowledge refinement, and corrective routing.
    """

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def grade_document(self, query: str, doc_content: str, metadata: Dict[str, Any]) -> Tuple[str, float, str, str]:
        """
        Grades an individual document against the query.
        Returns: (grade, score, rationale, refined_strip)
        """
        source_name = metadata.get("source_file", "Unknown Document")
        page_num = metadata.get("page", "?")

        grading_prompt = f"""You are an expert clinical retrieval evaluator (CRAG Grader).
Your job is to assess if the following document excerpt contains facts, recommendations, or criteria relevant to answering the clinical query.

Query: "{query}"

Document Source: {source_name} (Page {page_num})
Document Content:
\"\"\"{doc_content[:1500]}\"\"\"

Perform these tasks:
1. Grade the relevance as:
   - "CORRECT": Directly provides answers, diagnostic criteria, drug regimens, or clinical guidelines for the query.
   - "AMBIGUOUS": Discusses related concepts or general background but does not definitively answer the specific question.
   - "INCORRECT": Irrelevant, talking about a different disease/organ system, or mere table of contents/indexes without actionable information.
2. Provide a score between 0.0 (completely irrelevant) and 1.0 (perfectly relevant).
3. Extract only the 1-3 most salient clinical sentences (knowledge strip) that directly address the query. If INCORRECT, leave empty.

Respond ONLY with a JSON object:
{{
  "grade": "CORRECT" or "AMBIGUOUS" or "INCORRECT",
  "score": 0.0 to 1.0,
  "rationale": "Brief reason for this grade",
  "refined_strip": "Extracted salient sentences or empty string"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": grading_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            grade = data.get("grade", "AMBIGUOUS").upper()
            score = float(data.get("score", 0.5))
            rationale = data.get("rationale", "")
            refined_strip = data.get("refined_strip", "")
            return grade, score, rationale, refined_strip
        except Exception as e:
            return "AMBIGUOUS", 0.5, f"Grading error ({e})", doc_content[:200]

    def evaluate_retrieval(self, query: str, documents: List[Document]) -> CRAGDecision:
        """
        Evaluates a set of retrieved documents and determines the corrective action.
        """
        if not documents:
            return CRAGDecision(
                overall_status="INCORRECT",
                confidence_score=0.0,
                graded_documents=[],
                refined_context="",
                trigger_fallback=True,
                fallback_query=query,
                feedback="No documents retrieved from vector database."
            )

        graded_docs = []
        correct_count = 0
        ambiguous_count = 0
        total_score = 0.0
        refined_strips = []

        for doc in documents:
            grade, score, rationale, strip = self.grade_document(
                query=query,
                doc_content=doc.page_content,
                metadata=doc.metadata
            )
            total_score += score
            if grade == "CORRECT":
                correct_count += 1
            elif grade == "AMBIGUOUS":
                ambiguous_count += 1

            doc_repr = {
                "source_file": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page", 0),
                "content_preview": doc.page_content[:200] + "..."
            }
            graded_docs.append(GradedDocument(
                document=doc_repr,
                grade=grade,
                relevance_score=score,
                rationale=rationale,
                refined_strip=strip
            ))

            if grade in ("CORRECT", "AMBIGUOUS") and strip.strip():
                source_tag = f"[{doc.metadata.get('source_file', 'Doc')}, Page {doc.metadata.get('page', '?')}]"
                refined_strips.append(f"{source_tag}: {strip.strip()}")

        avg_score = total_score / len(documents) if documents else 0.0

        # CRAG Classification Logic
        # If at least one document is CORRECT or average score >= 0.65 -> CORRECT
        # If all documents are INCORRECT or avg_score < 0.35 -> INCORRECT (trigger corrective fallback)
        # Otherwise -> AMBIGUOUS
        if correct_count >= 1 or avg_score >= 0.60:
            overall_status = "CORRECT"
            trigger_fallback = False
            feedback = f"Retrieved {correct_count} highly relevant document(s). Knowledge refined successfully."
            fallback_query = ""
        elif ambiguous_count >= 1 and avg_score >= 0.35:
            overall_status = "AMBIGUOUS"
            trigger_fallback = False
            feedback = f"Retrieved partially relevant documents ({ambiguous_count} ambiguous). Refining context with cautionary confidence."
            fallback_query = ""
        else:
            overall_status = "INCORRECT"
            trigger_fallback = True
            feedback = "Retrieval failed relevance verification. Chunks are off-topic or irrelevant. Triggering CRAG corrective search."
            # Generate corrective search query
            fallback_query = self._generate_step_back_query(query)

        refined_context = "\n\n".join(refined_strips) if refined_strips else ""

        return CRAGDecision(
            overall_status=overall_status,
            confidence_score=round(avg_score, 3),
            graded_documents=graded_docs,
            refined_context=refined_context,
            trigger_fallback=trigger_fallback,
            fallback_query=fallback_query,
            feedback=feedback
        )

    def _generate_step_back_query(self, query: str) -> str:
        """
        Step-back prompting to formulate a broader or alternative clinical search query.
        """
        prompt = f"""You are a clinical query reformulation specialist in a CRAG pipeline.
The initial vector search for the query below returned irrelevant results.
Formulate a broader, alternative clinical query (step-back concept or core disease entity)
that might better locate relevant clinical guidelines in the medical database.

Initial Query: "{query}"
Corrective Reformulated Query:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60
            )
            return response.choices[0].message.content.strip().strip('"')
        except Exception:
            return query
