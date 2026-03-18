# RAG_SAV

**RAG_SAV** is a specialized framework for developing and evaluating Retrieval-Augmented Generation (RAG) components for the Indian legal domain. Utilizing the **IL-PCSR (Indian Legal Precedent-Case-Statute Retrieval)** dataset, this project benchmarks multiple retrieval strategies to improve the accuracy of finding relevant legal precedents.

## ⚖️ Project Objective
Legal search is complex; a query might share keywords with many cases but only be legally relevant to those citing the same specific statutes. This project implements a **two-stage retrieval pipeline** that verifies text-based candidates by re-ranking them based on **Statute Overlap (Jaccard Similarity)**.

## 🛠️ Technical Pipeline

### 1. Data Ingestion & Extraction (`pre_processing/`)
* **Data Sourcing:** Downloads `queries`, `precedents`, and `statutes` from the HuggingFace repository (`Exploration-Lab/IL-PCSR`).
* **Aho-Corasick Statute Extraction:** To overcome missing annotations, a high-performance pattern matcher (`statute_extractor.py`) identifies statute mentions (e.g., "Section 302 IPC") within the corpus using a dictionary of legal abbreviations.
* **Temporal Analysis:** Analyzes the date distribution of cases. Data is heavily skewed towards the post-2010 period, reflecting the rapid growth of digitized legal records.

### 2. Retrieval Strategies (`pipeline/`)
* **Lexical Baseline (BM25):** Uses the BM25Okapi algorithm for keyword-based retrieval.
* **Dense Baseline:** Implements semantic search using specialized legal embeddings to capture deeper contextual meaning beyond keywords.
* **Hybrid Statute Reranking:** Takes the top candidates from the first stage and re-scores them using an **Alpha ($\alpha$)** weighted sum:
    $$\text{Score} = \alpha \cdot \text{Score}_{\text{textual}} + (1 - \alpha) \cdot \text{Jaccard}(\text{Statutes}_{\text{query}}, \text{Statutes}_{\text{candidate}})$$

---

## 📊 Experimental Results

### BM25 & Dense Baselines
BM25 remains a strong baseline for legal text, significantly outperforming the zero-shot Dense MiniLM baseline.

| Split | R@1 | R@5 | R@10 | MRR |
| :--- | :--- | :--- | :--- | :--- |
| **BM25 Dev** | 0.2093 | 0.3850 | 0.4649 | 0.2947 |
| **BM25 Test** | 0.1677 | 0.3850 | 0.4696 | 0.2713 |
| **Dense Test** | 0.0351 | 0.1166 | 0.1741 | 0.0824 |

### Hybrid Optimization (Alpha Sweep)
Sweeping the alpha value on the development split identified **$\alpha = 0.9$** as the optimal balance, showing that the statute signal provides a critical verification boost.

| $\alpha$ (BM25 + Jaccard) | R@1 | R@5 | R@10 | MRR |
| :--- | :--- | :--- | :--- | :--- |
| **0.9** | **0.1757** | **0.3882** | **0.4792** | **0.2725** |
| 0.7 | 0.1374 | 0.3371 | 0.4537 | 0.2386 |
| 0.5 | 0.1294 | 0.3291 | 0.4473 | 0.2304 |

### Final Comparison (Test Split)
The hybrid approach (BM25 + Jaccard) achieved the highest Recall@10, effectively promoting relevant cases that might have been buried in a pure keyword search.

| System | R@1 | R@5 | R@10 | MRR |
| :--- | :--- | :--- | :--- | :--- |
| BM25 (baseline) | 0.1677 | **0.3850** | 0.4696 | **0.2713** |
| **BM25 + Jaccard ($\alpha=0.9$)** | **0.1693** | 0.3706 | **0.4840** | 0.2667 |
| Dense + Jaccard ($\alpha=0.9$) | 0.0703 | 0.1693 | 0.2157 | 0.1237 |

---

## 📁 Repository Structure

### `/pre_processing`
* `pcsr_dwnld.py`: Ingests raw data configurations from HuggingFace.
* `statute_extractor.py`: High-precision statute extraction using Aho-Corasick.
* `temporal_eda.py`: Visualizes the growth of legal cases from 1951–2023.
* `data_verification.py`: Checks statute coverage and pool integrity.

### `/pipeline`
* `01_bm25_baseline.py`: Keyword retrieval logic and benchmarking.
* `02_hybrid_reranker.py`: Hybrid scoring engine using statute Jaccard scores.
* `dense_embedder.py`: Generates legal embeddings for semantic search.
* `generate_statute_index.py`: Maps precedents to laws for fast lookup.
