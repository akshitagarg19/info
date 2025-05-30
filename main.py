# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import openai

# ✅ HARD-CODED OpenAI API Key (for testing only)
openai.api_key = "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"

# Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS for all origins and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request payload model
class SearchRequest(BaseModel):
    docs: List[str]
    query: str

# ✅ Call OpenAI Embedding API
def get_embeddings(texts: List[str]) -> List[List[float]]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [record["embedding"] for record in response.data]

# ✅ Cosine similarity
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# ✅ POST endpoint
@app.post("/api/v1/embedding-search")
async def embedding_search(payload: SearchRequest):
    if not payload.docs or not payload.query:
        return {"matches": [], "error": "Empty docs or query."}

    # Combine docs + query and get embeddings
    all_texts = payload.docs + [payload.query]
    embeddings = get_embeddings(all_texts)

    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    # Compute similarity and rank
    similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]
    top_docs = [payload.docs[i] for i in top_indices]

    return {"matches": top_docs}
