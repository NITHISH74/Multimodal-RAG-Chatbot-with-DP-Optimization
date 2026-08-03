"""
Evaluation harness for the Multi-Model RAG pipeline.

Public entry points:
    - eval.run_eval        — end-to-end metrics against the golden core
    - eval.ablation        — vector-only vs hybrid vs hybrid+rerank vs hybrid+rerank+knapsack
    - eval.judge            — LLM-as-judge for answer faithfulness / relevance
    - eval.metrics          — pure-Python Recall@k, HitRate@k, MRR, NDCG@k
    - eval.sim              — shared cosine similarity
"""
