"""
Complete Medical RAG Orchestrator.
Integrates:
- Qdrant Vector Database
- Input Guardrail (Security, Domain Relevance, Safety)
- Query Transformer (Rewriting, Sub-Query Decomposition, HyDE)
- Corrective RAG (CRAG: Document Relevance Grading, Knowledge Refinement, Step-Back Fallback)
- Output Guardrail (Faithfulness, Hallucination Verification, Clinical Safety, Mandatory Disclaimer)
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document

# Dynamic imports from current package
CURRENT_DIR = Path(__file__).resolve().parent
RAG_DIR = CURRENT_DIR.parent
BASE_DIR = RAG_DIR.parent

# Add path for imports
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from query_transformer import QueryTransformer, QueryTransformResult
from guardrails import InputGuardrail, OutputGuardrail, InputGuardrailResult, OutputGuardrailResult
from crag import CorrectiveRAG, CRAGDecision

# Load environment
env_path = BASE_DIR / ".env"
if not env_path.exists():
    env_path = RAG_DIR / ".env"
load_dotenv(dotenv_path=env_path)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in environment or .env file")

QDRANT_PATH = str(RAG_DIR / "qdrant_db")
COLLECTION_NAME = "medical_documents"


class RAGResponse(BaseModel):
    query: str
    final_answer: str
    input_guardrail: InputGuardrailResult
    query_transform: Optional[QueryTransformResult] = None
    crag_decision: Optional[CRAGDecision] = None
    crag_fallback_triggered: bool = False
    output_guardrail: Optional[OutputGuardrailResult] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    execution_trace: List[str] = Field(default_factory=list)


class MedicalRAGPipeline:
    def __init__(
        self,
        openai_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = COLLECTION_NAME,
        qdrant_path: str = QDRANT_PATH,
    ):
        self.client = OpenAI()
        self.model = openai_model
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path

        # 1. Initialize Embeddings & Vector Store
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.qdrant_client = QdrantClient(path=self.qdrant_path)

        if not self.qdrant_client.collection_exists(self.collection_name):
            raise RuntimeError(
                f"Collection '{self.collection_name}' not found at '{self.qdrant_path}'. "
                "Please run ingestion/indexing.py first!"
            )

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

        # 2. Optimization Components
        self.input_guardrail = InputGuardrail(client=self.client, model=self.model)
        self.query_transformer = QueryTransformer(client=self.client, model=self.model)
        self.crag = CorrectiveRAG(client=self.client, model=self.model)
        self.output_guardrail = OutputGuardrail(client=self.client, model=self.model)

    def retrieve_documents(self, queries: List[str], top_k: int = 4) -> List[Document]:
        """
        Retrieves documents from Qdrant across multiple queries with deduplication.
        """
        all_docs = []
        seen_keys = set()

        for q in queries:
            try:
                results = self.vector_store.similarity_search(query=q, k=top_k)
                for doc in results:
                    # Key based on source file, page, and chunk content prefix
                    doc_key = (
                        doc.metadata.get("source_file", ""),
                        doc.metadata.get("page", 0),
                        doc.page_content[:100]
                    )
                    if doc_key not in seen_keys:
                        seen_keys.add(doc_key)
                        all_docs.append(doc)
            except Exception as e:
                print(f"[Warning] Retrieval error for query '{q}': {e}")

        return all_docs[:8]  # Limit to top 8 distinct documents for grading

    def generate_synthesis(self, query: str, context: str, crag_status: str) -> str:
        """
        Synthesizes a medically accurate, cited answer using refined clinical context.
        """
        system_instruction = """You are an evidence-based clinical decision support assistant.
Answer the user's medical query based ONLY on the provided clinical guideline context.
Requirements:
1. Directly cite the specific source guideline and page number for every recommendation (e.g. [hypertension_full.pdf, Page 12]).
2. Be clinically precise regarding drug names, first-line options, contraindications, and dosages when present.
3. If the context is ambiguous or partial, explicitly note the limitations in the guidelines.
4. Structure your response with clear clinical headings (e.g., Overview, Guideline Recommendations, Key Considerations).
"""

        user_content = f"""Query: {query}
CRAG Confidence Status: {crag_status}

Refined Clinical Guideline Context:
{context}

Please provide an authoritative clinical response based on the above guidelines:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()

    def query(self, user_query: str, verbose: bool = True) -> RAGResponse:
        trace = []
        def log(step: str):
            trace.append(step)
            if verbose:
                print(step)

        log(f"\n{'='*70}\n[RAG PIPELINE] Processing Query: '{user_query}'\n{'='*70}")

        # --- STEP 1: INPUT GUARDRAIL ---
        log("\n[Step 1] Running Input Guardrail (Security, Domain & Safety)...")
        input_eval = self.input_guardrail.check(user_query)

        if not input_eval.is_allowed:
            log(f"-> [BLOCKED by Input Guardrail] Category: {input_eval.violation_category}")
            log(f"   Reason: {input_eval.reason}")
            refusal_response = (
                f"**Request Blocked by Clinical Guardrail**\n\n"
                f"**Category:** {input_eval.violation_category.upper()}\n"
                f"**Reason:** {input_eval.reason}\n\n"
                f"This system is strictly dedicated to assisting with evidence-based medical inquiries, "
                f"clinical guidelines, pharmacotherapy, and health conditions. Adversarial inputs, prompt overrides, "
                f"and off-topic non-medical questions are rejected."
            )
            return RAGResponse(
                query=user_query,
                final_answer=refusal_response,
                input_guardrail=input_eval,
                execution_trace=trace
            )
        log("-> [PASSED] Input is safe and within the clinical guideline scope.")

        # --- STEP 2: QUERY TRANSFORMER ---
        log("\n[Step 2] Applying Query Transformer (Rewriting, Sub-Query Decomposition & HyDE)...")
        transformed = self.query_transformer.transform(user_query, mode="all")
        log(f"   Original Query  : {transformed.original_query}")
        log(f"   Rewritten Query : {transformed.rewritten_query}")
        log(f"   Sub-Queries ({len(transformed.sub_queries)}):")
        for sq in transformed.sub_queries:
            log(f"     - {sq}")

        # --- STEP 3: RETRIEVAL FROM QDRANT ---
        log("\n[Step 3] Retrieving Candidate Chunks from Qdrant Vector Database...")
        # Use rewritten query + sub-queries for diverse retrieval
        search_queries = [transformed.rewritten_query] + transformed.sub_queries[:2]
        candidate_docs = self.retrieve_documents(search_queries, top_k=4)
        log(f"-> Retrieved {len(candidate_docs)} distinct candidate document chunks.")

        # --- STEP 4: CORRECTIVE RAG (CRAG) EVALUATION ---
        log("\n[Step 4] Running Corrective RAG (CRAG Evaluator & Knowledge Refiner)...")
        crag_result = self.crag.evaluate_retrieval(user_query, candidate_docs)
        log(f"-> CRAG Overall Status : {crag_result.overall_status} (Confidence: {crag_result.confidence_score})")
        log(f"   Feedback           : {crag_result.feedback}")

        for i, gd in enumerate(crag_result.graded_documents, 1):
            source = gd.document.get("source_file", "unknown")
            page = gd.document.get("page", 0)
            log(f"   Chunk #{i}: [{source} p.{page}] Grade={gd.grade} (Score: {gd.relevance_score}) - {gd.rationale}")

        # Handle CRAG Fallback if retrieval failed
        fallback_triggered = False
        if crag_result.trigger_fallback:
            fallback_triggered = True
            log(f"\n[CRAG Fallback Triggered] Attempting secondary search with reformulated query: '{crag_result.fallback_query}'")
            secondary_docs = self.retrieve_documents([crag_result.fallback_query], top_k=4)
            secondary_crag = self.crag.evaluate_retrieval(crag_result.fallback_query, secondary_docs)
            log(f"-> Secondary CRAG Status : {secondary_crag.overall_status} (Confidence: {secondary_crag.confidence_score})")

            if secondary_crag.overall_status in ("CORRECT", "AMBIGUOUS"):
                crag_result = secondary_crag
                candidate_docs = secondary_docs
                log("-> Secondary search successfully recovered relevant clinical knowledge!")
            else:
                log("-> Secondary search also found no matching clinical guidelines in the database.")
                safe_fallback_answer = (
                    f"**Clinical Guidelines Notice: No Documented Information Found**\n\n"
                    f"The Corrective RAG (CRAG) evaluator assessed all retrieved document chunks from the "
                    f"current clinical guidelines database against your query (`\"{user_query}\"`) and determined that "
                    f"the collection does not contain authoritative evidence or guidelines on this specific clinical question.\n\n"
                    f"**Topics covered in this database include:**\n"
                    f"- Antimicrobial guidelines & prescribing\n"
                    f"- Acute respiratory infections (ARI) in children\n"
                    f"- Ayurveda principles & remedies\n"
                    f"- Cardiovascular diseases & management\n"
                    f"- Dengue clinical protocols\n"
                    f"- General surgery guidelines\n"
                    f"- Hypertension management & quick reference\n"
                    f"- Malaria treatment protocols\n"
                    f"- Neurology clinical practice\n"
                    f"- Obstetrics & Gynaecology guidelines\n\n"
                    f"To avoid generating hallucinated or unverified medical advice, this system refrains from answering without grounded context."
                )
                output_eval = self.output_guardrail.check(
                    query=user_query,
                    context="No relevant context",
                    draft_response=safe_fallback_answer
                )
                return RAGResponse(
                    query=user_query,
                    final_answer=output_eval.finalized_response,
                    input_guardrail=input_eval,
                    query_transform=transformed,
                    crag_decision=crag_result,
                    crag_fallback_triggered=fallback_triggered,
                    output_guardrail=output_eval,
                    sources=[],
                    execution_trace=trace
                )

        # --- STEP 5: GROUNDED SYNTHESIS ---
        log("\n[Step 5] Synthesizing Medical Response using Refined Knowledge Strips...")
        context_to_use = crag_result.refined_context if crag_result.refined_context else "\n\n".join([d.page_content for d in candidate_docs[:3]])
        draft_answer = self.generate_synthesis(user_query, context_to_use, crag_result.overall_status)

        # --- STEP 6: OUTPUT GUARDRAIL ---
        log("\n[Step 6] Running Output Guardrail (Factuality Verification, Safety & Disclaimer)...")
        output_eval = self.output_guardrail.check(
            query=user_query,
            context=context_to_use,
            draft_response=draft_answer
        )
        log(f"-> Output Guardrail Result: is_safe={output_eval.is_safe}, hallucination_detected={output_eval.hallucination_detected}")
        log(f"   Reason: {output_eval.reason}")

        # Compile sources
        unique_sources = []
        for gd in crag_result.graded_documents:
            if gd.grade in ("CORRECT", "AMBIGUOUS"):
                unique_sources.append({
                    "source_file": gd.document.get("source_file"),
                    "page": gd.document.get("page"),
                    "relevance_grade": gd.grade,
                    "confidence": gd.relevance_score
                })

        log("\n[Pipeline Complete] Delivering Grounded Clinical Response.")

        return RAGResponse(
            query=user_query,
            final_answer=output_eval.finalized_response,
            input_guardrail=input_eval,
            query_transform=transformed,
            crag_decision=crag_result,
            crag_fallback_triggered=fallback_triggered,
            output_guardrail=output_eval,
            sources=unique_sources,
            execution_trace=trace
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clinical RAG with Optimization, Guardrails & CRAG")
    parser.add_argument("--query", "-q", type=str, default=None, help="Query to run")
    args = parser.parse_args()

    pipeline = MedicalRAGPipeline()

    if args.query:
        res = pipeline.query(args.query, verbose=True)
        print("\n" + "="*80)
        print("FINAL CLINICAL RESPONSE:")
        print("="*80)
        print(res.final_answer)
    else:
        # Default interactive test
        test_query = "What is the recommended first-line treatment for uncomplicated malaria and in pregnancy?"
        print(f"\nRunning demonstration query: '{test_query}'\n")
        res = pipeline.query(test_query, verbose=True)
        print("\n" + "="*80)
        print("FINAL CLINICAL RESPONSE:")
        print("="*80)
        print(res.final_answer)


if __name__ == "__main__":
    main()
