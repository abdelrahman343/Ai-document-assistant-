# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July) 2026

> 🏆 This repository is my official submission for the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Abdelrahman Sameh                    |
| Project Name     | Document Assistant                   |
| GitHub Username  | [@abdelrahman343](https://github.com/abdelrahman343) |
| Challenge Batch  | June–July 2026                       |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en)                         |

---

# 📖 Project Overview

**Document Assistant** is a Retrieval-Augmented Generation (RAG) application that lets you upload documents and ask questions about them in plain language. Each document is indexed independently — nothing gets merged or searched across files — so answers and citations always come from the specific document you're chatting with.

Under the hood it combines dense (embedding-based) and sparse (keyword-based) retrieval using Reciprocal Rank Fusion, expands ambiguous questions into multiple search queries before retrieving, and generates answers strictly grounded in the retrieved context, citing the exact file and page for every claim.

---

# ✨ Features

* **Multi-format document support** — PDF, DOCX, PPTX, and TXT, including table content extraction and grouped-shape handling in slides.
* **Per-document isolation** — every uploaded file gets its own independent index; questions never pull answers from the wrong document.
* **Hybrid retrieval** — BM25 (keyword) and FAISS (semantic) search fused with Reciprocal Rank Fusion, with a minimum relevance threshold so unrelated questions don't return irrelevant chunks.
* **Query expansion** — ambiguous questions are automatically rewritten into multiple phrasings to improve recall (skipped for short, already-specific questions to save latency).
* **Grounded answers with citations** — every answer is generated strictly from retrieved context and shows the exact file + page it came from.
* **Document summaries & suggested questions** — one-click overview of any uploaded document, plus auto-generated starter questions.
* **Greeting detection** — casual messages ("hi", "thanks") get a friendly direct reply instead of an unnecessary retrieval pass.
* **Performance-optimized** — singleton model/client caching, disk-persisted indexes (re-uploading a previously seen file skips re-embedding entirely), parallel indexing for multi-file uploads, and answer caching for repeated questions.

---

# 🛠️ Technologies Used

* **Frontend**: Streamlit
* **LLM**: Groq (Llama 3.3 70B)
* **Embeddings**: Sentence-Transformers (`BAAI/bge-small-en-v1.5`)
* **Vector Search**: FAISS
* **Keyword Search**: BM25 (`rank_bm25`)
* **Document Parsing**: PyMuPDF (PDF), python-docx, python-pptx
* **Language**: Python

---

# ⚙️ Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key:

```
GROQ_API_KEY=your_key_here
```

---

# 🚀 Usage

Run the app:

```bash
streamlit run app.py
```

Then:

1. Upload one or more documents in the sidebar (indexing happens automatically).
2. Pick a document from the dropdown to chat with.
3. Ask questions, generate a summary, or browse suggested questions — all scoped to that document.

---

---

# 📈 Results

During manual testing, the assistant correctly answered questions grounded in the uploaded document — in both English and Arabic — and appropriately declined to answer unrelated questions instead of hallucinating.

---

# 🔮 Future Improvements

* LLM-based intent classification instead of keyword matching for summary/greeting detection.
* Retry/backoff on LLM API calls for better reliability under rate limits.
* Authentication and per-user usage limits for shared deployments.
* Tunable chunk size validated against real documents rather than a fixed default.
* Cache eviction policy for the on-disk index cache.

---

# 📚 About the Challenge

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the challenge encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
