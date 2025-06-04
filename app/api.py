# app/api.py
from fastapi import FastAPI, HTTPException
from .rag_chain import build_chain  # we'll stub this for now

app = FastAPI(title="Insurance-RAG")

# Build the retrieval-augmented chain once at startup
qa_chain = build_chain()

@app.get("/healthz")
def healthz() -> dict:
    """Simple liveness probe for Docker & k8s."""
    return {"status": "ok"}

@app.post("/query")
def query(q: str) -> dict:
    """
    Ask a question about your insurance PDF.

    Body param: { "q": "Does the plan cover emergency dental?" }
    """
    try:
        result = qa_chain.invoke({"query": q})
        return {
            "answer": result["result"],
            "sources": [
                {
                    "page": d.metadata.get("page"),
                    "snippet": d.page_content[:160] + "…",
                }
                for d in result["source_documents"]
            ],
        }
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))


