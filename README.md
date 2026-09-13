# 🎓 RAG System for Corporate Knowledge Retrieval

> **Bachelor Thesis** — Università degli Studi di Brescia  
> **Author:** Mattia Massolari  
> **Year:** 2025

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system designed to answer questions about corporate knowledge stored in internal ticketing/progress-tracking data. The system was developed as part of my Bachelor's degree thesis and demonstrates how LLMs can be augmented with domain-specific context to provide accurate, grounded answers from proprietary datasets.

The RAG pipeline ingests, cleans, and indexes **thousands of internal support messages** from a corporate environment, then uses vector similarity search to retrieve relevant context before generating answers via multiple LLM backends.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     DATA INGESTION                      │
│  CSV (Progress + Messages) → Cleaning → JSON Export     │
│  • HTML stripping (BeautifulSoup)                       │
│  • Deduplication (TF-IDF cosine similarity ≥ 0.95)      │
│  • Grouping messages by progress thread                 │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   TEXT PROCESSING                        │
│  Chunking (RecursiveCharacterTextSplitter)               │
│  • Tested: 300/50, 500/150, 800/100 (size/overlap)      │
│  Embedding (OpenAI Embeddings)                           │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   VECTOR STORE                          │
│  FAISS Index (Facebook AI Similarity Search)             │
│  • Similarity Search & MMR retrieval strategies          │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│               GENERATION & EVALUATION                   │
│  Prompt Template → LLM → Answer                         │
│  • True/False and open-ended question formats            │
│  • Semantic similarity scoring (cosine)                  │
│  • QAGS-based factual consistency validation             │
└─────────────────────────────────────────────────────────┘
```

## 🧪 Models Benchmarked

The system was evaluated across **7 different LLMs** to compare answer accuracy:

| Model            | Provider   | Type        |
| ---------------- | ---------- | ----------- |
| GPT-4.1          | OpenAI     | Proprietary |
| GPT-4-Turbo      | OpenAI     | Proprietary |
| GPT-4            | OpenAI     | Proprietary |
| GPT-3.5-Turbo    | OpenAI     | Proprietary |
| Gemini 2.5 Flash | Google     | Proprietary |
| Mistral Tiny     | Mistral AI | Open-weight |
| Meta-Llama       | Meta       | Open-weight |

## 📊 Key Results

Evaluation was performed using **5 test sets** with both **True/False** and **open-ended** question formats.

### True/False Questions (Similarity Retrieval)

| Model             | Best Accuracy |
| ----------------- | ------------- |
| **GPT-3.5-Turbo** | **86.67%**    |
| **GPT-4-Turbo**   | **86.67%**    |
| **GPT-4**         | **86.67%**    |
| **Mistral Tiny**  | **86.67%**    |
| GPT-4.1           | 73.33%        |
| Gemini 2.5 Flash  | 73.33%        |

### Open-ended Questions

Evaluated using **semantic similarity** against expected answers and validated with a **QAGS (Question-Answer Generation for Summarization)** metric using a fine-tuned NLI model for factual consistency scoring.

### Chunk Size Impact

| Chunk Config            | Avg. Accuracy         |
| ----------------------- | --------------------- |
| 500 chars / 150 overlap | **Best balance**      |
| 800 chars / 100 overlap | Good for long threads |
| 300 chars / 50 overlap  | Lower accuracy        |

## 🔧 Tech Stack

| Category            | Technologies                                     |
| ------------------- | ------------------------------------------------ |
| **Language**        | Python 3.11                                      |
| **LLM Framework**   | LangChain                                        |
| **Vector Store**    | FAISS (faiss-cpu)                                |
| **Embeddings**      | OpenAI Embeddings                                |
| **LLM Providers**   | OpenAI, Google Gemini, Mistral AI                |
| **NLP / ML**        | Hugging Face Transformers, scikit-learn, PyTorch |
| **Data Processing** | Pandas, BeautifulSoup, TF-IDF Vectorizer         |
| **Visualization**   | Matplotlib, Seaborn                              |
| **Environment**     | Google Colab (with GPU support for local models) |

## 📂 Project Structure

```
BachelorThesis/
├── RAG/
│   ├── RagSystem.ipynb        # Main RAG pipeline (data ingestion, indexing,
│   │                          # multi-model evaluation, True/False tests)
│   └── RagSystem1.ipynb       # Extended evaluation with open-ended questions,
│                              # QAGS validation, HuggingFace NLI model
├── Stats/
│   ├── QUESTION/              # Test question sets (5 test files)
│   ├── ANSWER/                # Expected answer sets (5 answer files)
│   ├── TEST_RESULTS_SIMILARITY/   # Results using similarity search retrieval
│   ├── TEST_RESULTS_MMR/          # Results using MMR retrieval
│   ├── RESULTS_CESARE/            # Cross-validation results
│   ├── results_summary.csv        # Aggregated model accuracy results
│   ├── chunk_result_summary.csv   # Chunk size ablation study
│   └── tab_stats.csv              # Detailed statistical analysis
├── Thesis/
│   ├── Tesi_Triennale_Massolari_740458_noRingraziamenti.pdf  # Full thesis (PDF)
│   └── Tesi_MAT_2025.pptx                                    # Defense slides
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- API keys for: OpenAI, Google Gemini, Mistral AI
- (Optional) Hugging Face token for gated models

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/BachelorThesis.git
cd BachelorThesis
pip install -r requirements.txt
```

### Usage

The notebooks are designed to run on **Google Colab** with Google Drive integration. To run locally:

1. Update file paths in the notebooks to point to your local data directory
2. Set your API keys as environment variables
3. Run the notebooks with Jupyter

```bash
jupyter notebook RAG/RagSystem.ipynb
```

> ⚠️ **Note:** The raw corporate data (CSV files) is not included in this repository for privacy reasons. The processed output and evaluation results are available in the `Stats/` directory.

## 📚 Methodology Highlights

- **Data Cleaning:** HTML stripping, non-ASCII removal, whitespace normalization, and near-duplicate removal using TF-IDF cosine similarity (threshold ≥ 0.95)
- **Chunking Strategy:** Systematic evaluation of 3 chunk configurations to find optimal text segmentation
- **Retrieval Comparison:** Side-by-side evaluation of Similarity Search vs. MMR (Maximal Marginal Relevance) retrieval strategies
- **Multi-Model Benchmarking:** Consistent test protocol across 7 LLMs with statistical significance analysis
- **QAGS Validation:** Factual consistency scoring using NLI (Natural Language Inference) transformer models for open-ended answer evaluation

## 📈 Dataset Statistics

| Metric                   | Value                  |
| ------------------------ | ---------------------- |
| Total progress threads   | 574                    |
| Total messages           | 8,669                  |
| Unique authors           | 49                     |
| Avg. messages per thread | 15.1                   |
| Avg. message length      | 922 chars (~138 words) |

## 📄 License

This project was developed for academic purposes as part of a Bachelor's thesis at the University of Brescia.
