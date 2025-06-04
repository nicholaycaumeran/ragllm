## RAG-LLM on Insurance Policy: A production‑ready Retrieval‑Augmented Generation pipeline for answering questions about insurance policy documents.

# This repository contains a complete reference implementation of a RAG (Retrieval‑Augmented Generation) system built with LangChain 0.1.x and evaluated with Ragas 0.2.15.  The pipeline ingests an insurance policy PDF, chunks & indexes it with FAISS, and serves a FastAPI endpoint that answers free‑text questions grounded in the source document.

# Key components
Chunking & Embeddings: Convert policy text into dense vectors
- LangChain + OpenAI text‑embedding‑3‑small

Vector store: k‑NN similarity search
- FAISS (in‑memory)

LLM: Draft answers & cite sources
- OpenAI gpt‑4o‑mini (configurable)

Web API: /query?q=… & Swagger UI
- FastAPI + Uvicorn

Automated evaluation
- Ragas context‑precision & faithfulness

Container: Reproducible prod image
- Python 3.12‑slim Dockerfile

CI/CD: Build, test, push Docker image
- GitHub

##  Quick Start (local)
1. clone & enter repo
$ git clone https://github.com/nicholaycaumeran/ragllm.git
$ cd ragllm

2. create Python 3.12 venv
$ python -m venv .venv && source .venv/Scripts/activate

3. install deps (CPU‑only)
$ pip install -r requirements.txt

4. add your OpenAI key
$ echo "OPENAI_API_KEY=sk‑xxx" > .env

5. run automated quality gate
$ python -m pytest -q          # fails if precision/faithfulness < 0.70

6. launch API server
$ uvicorn app.api:app --reload --port 8080
note: browse → http://localhost:8080/docs

## D0cker:
1. build image (~750 MB)
$ docker build -t rag-insurance:latest .

2. run container
$ docker run --rm -p 8080:8080 --env-file .env rag-insurance:latest
