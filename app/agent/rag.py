from __future__ import annotations

from typing import Tuple, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from .kb import load_kb_text


def build_vectorstore(kb_path: str) -> FAISS:
    text = load_kb_text(kb_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)


def retrieve(vs: FAISS, query: str, k: int = 4) -> Tuple[str, List[str]]:
    docs = vs.similarity_search(query, k=k)
    context = "\n\n".join(d.page_content for d in docs)
    return context, [d.page_content for d in docs]
