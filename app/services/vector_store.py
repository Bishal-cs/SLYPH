"""
VECTOR STORE SERVICE MODULE
===========================

This service builds and queries the FAISS vector index used for context retrieval.
Learning data (database/learning_data/*.txt) and past chats (database/chats_data/*.json) 
are loaded at startup and, split into chunks, embedded with HuggingFace, and stored in FAISS.
When the user asks a question we embed it and retrieve the k most similar chunks; only
those chunks are sent to the LLM, so the context is limited to the most relevant information, 
improving response quality and reducing token usage.

LIFECYCLE:
 - create_vector_store(): Load all .txt and .json, chunk, embed, build FAISS, save to disk.
   Called once at startup. Restart the server after adding new .txt files so they are included.
 - get_retriever(k): Return a retriever that fetches k nearest chunks for a query string.
 - save_vector_store(): Write the current FAISS index to database/vector_store/ (called after create).

Embeddings run locally (sentence-transformers); no extra API key. Groq and Realtime services
call get_retriever() for every request to get context.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    LEARNING_DATA_DIR,
    CHATS_DATA_DIR,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger("S.Y.L.P.H")
