"""Hybrid-retrieval RAG service with a measured eval harness.

Dense (pgvector) + sparse (Postgres full-text) retrieval fused with Reciprocal
Rank Fusion, an LLM reranker, and citation-grounded answers that abstain on weak
retrieval. Built on the Claude API with structured tool-calling.
"""
