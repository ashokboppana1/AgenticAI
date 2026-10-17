"""
Query Transformer Module for Medical RAG.
Implements:
1. Query Rewriting / Clinical Expansion: Reformulates colloquial or vague questions into standard medical search terms.
2. Multi-Query Decomposition: Splits multi-faceted medical queries into focused sub-queries for broader coverage.
3. HyDE (Hypothetical Document Embeddings): Generates a hypothetical clinical guideline snippet to align vector representations.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI


class QueryTransformResult(BaseModel):
    original_query: str
    rewritten_query: str
    sub_queries: List[str] = Field(default_factory=list)
    hyde_passage: str = ""
    all_search_queries: List[str] = Field(default_factory=list)


class QueryTransformer:
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def rewrite_query(self, query: str) -> str:
        """
        Rewrites the query using professional medical terminology and synonyms
        found in clinical guidelines.
        """
        prompt = f"""You are an expert medical search optimizer. 
Rewrite the following user query into a concise, clinically accurate search query suitable for searching clinical practice guidelines and medical textbooks.
Include relevant medical terminology, standard drug classes, or diagnostic categories.
Do not add conversational fluff. Return ONLY the rewritten query.

User Query: "{query}"
Rewritten Medical Search Query:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )
        return response.choices[0].message.content.strip().strip('"')

    def decompose_query(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Decomposes complex, multi-part clinical queries into distinct sub-queries.
        """
        prompt = f"""You are a clinical query decomposition assistant.
Break down the following complex medical query into {num_queries} specific, distinct sub-queries to retrieve all necessary guideline context (e.g. diagnosis, first-line treatment, contraindications, or risk factors).
Return each sub-query on a new line starting with a dash (- ). Do not include numbering or preamble.

Query: "{query}"
Sub-queries:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )
        lines = response.choices[0].message.content.strip().split("\n")
        sub_queries = []
        for line in lines:
            cleaned = line.strip().lstrip("-").lstrip("0123456789.").strip()
            if cleaned:
                sub_queries.append(cleaned)
        return sub_queries if sub_queries else [query]

    def generate_hyde(self, query: str) -> str:
        """
        Hypothetical Document Embeddings (HyDE):
        Generates a hypothetical clinical guideline excerpt that directly answers the query.
        The vector embedding of this passage bridges the semantic gap between questions and guideline documents.
        """
        prompt = f"""You are a clinical textbook writer. Write a short, authoritative clinical guideline passage (3-5 sentences) that answers the following query.
Use standard medical textbook style, citing typical first-line recommendations, criteria, or dosages.

Query: "{query}"
Clinical Guideline Excerpt:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()

    def transform(self, query: str, mode: str = "all") -> QueryTransformResult:
        """
        Executes query transformation pipeline.
        mode can be: 'rewrite', 'decompose', 'hyde', or 'all'.
        """
        rewritten = self.rewrite_query(query)
        sub_queries = []
        hyde_doc = ""
        search_queries = [query, rewritten]

        if mode in ("decompose", "all"):
            sub_queries = self.decompose_query(query)
            search_queries.extend(sub_queries)

        if mode in ("hyde", "all"):
            hyde_doc = self.generate_hyde(query)

        # Deduplicate search queries while preserving order
        unique_search_queries = []
        seen = set()
        for q in search_queries:
            normalized = q.lower().strip()
            if normalized not in seen and len(normalized) > 0:
                seen.add(normalized)
                unique_search_queries.append(q)

        return QueryTransformResult(
            original_query=query,
            rewritten_query=rewritten,
            sub_queries=sub_queries,
            hyde_passage=hyde_doc,
            all_search_queries=unique_search_queries
        )
