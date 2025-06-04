# tests/test_ragas.py  –  ragas 0.2.15 compatible
import sys, pathlib, yaml, pytest, pandas as pd, numpy as np
from datasets import Dataset
# ── make `app` importable ────────────────────────────────────────────────
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
# ─────────────────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness
from app.rag_chain import build_chain

@pytest.fixture(scope="session")
def rag_chain():
    """Build the Retrieval-QA chain only once for the test session."""
    return build_chain()

@pytest.fixture(scope="session")
def eval_df():
    """Load YAML evaluation set → DataFrame with required column names."""
    rows = yaml.safe_load(open("tests/dataset.yaml", encoding="utf-8"))
    return (
        pd.DataFrame(rows)
          .rename(columns={"answer": "ground_truth"})
    )

def generate_answers_and_contexts(rag_chain, questions):
    """Generate answers and contexts using the RAG chain."""
    answers = []
    contexts = []
    
    for question in questions:
        # Get the answer from your RAG chain
        response = rag_chain.invoke({"query": question})
        
        # Extract answer and source documents
        answer = response.get("result", "")
        source_docs = response.get("source_documents", [])
        
        # Extract contexts from source documents
        context = [doc.page_content for doc in source_docs]
        
        answers.append(answer)
        contexts.append(context)
    
    return answers, contexts

def test_ragas_scores(rag_chain, eval_df):
    """Fail CI if average precision / faithfulness drop below 0.70."""
    
    # Generate answers and contexts using your RAG chain
    questions = eval_df["question"].tolist()
    answers, contexts = generate_answers_and_contexts(rag_chain, questions)
    
    # Prepare the dataset for RAGAS evaluation
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": eval_df["ground_truth"].tolist()
    })
    
    # Evaluate using RAGAS
    result = evaluate(
        dataset=eval_dataset,
        metrics=[context_precision, faithfulness],
    )
    
    # Extract scores and calculate means - RAGAS 0.2.15 returns lists
    avg_context_precision = np.mean(result["context_precision"])
    avg_faithfulness = np.mean(result["faithfulness"])
    
    # Check if scores meet the threshold
    assert avg_context_precision > 0.70, f"Context precision {avg_context_precision:.3f} below threshold"
    assert avg_faithfulness > 0.70, f"Faithfulness {avg_faithfulness:.3f} below threshold"
    
    print(f"Context Precision: {avg_context_precision:.3f}")
    print(f"Faithfulness: {avg_faithfulness:.3f}")
    print(f"Individual Context Precision scores: {result['context_precision']}")
    print(f"Individual Faithfulness scores: {result['faithfulness']}")
