# app/rag_chain.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

def build_chain() -> RetrievalQA:
    """Build RetrievalQA with FAISS + GPT-4o-mini."""

    # 1. Load & chunk every PDF in data/
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    for pdf in Path("data").glob("*.pdf"):
        pages = PyPDFLoader(str(pdf)).load()
        docs += splitter.split_documents(pages)

    # 2. Embed and store in FAISS (in-memory)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb  = FAISS.from_documents(docs, embeddings)

    # 3. Chat model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # 4. Retrieval-QA pipeline
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True,
    )

