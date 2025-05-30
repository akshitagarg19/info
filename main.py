# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import openai

# Set your OpenAI API key here or export OPENAI_API_KEY in your environment
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

app = FastAPI()

# Enable CORS for all origins and allow OPTIONS and POST methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["OPTIONS", "POST"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    docs: List[str]
    query: str

def embed_texts(texts: List[str]) -> List[List[float]]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item["embedding"] for item in response["data"]]

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.post("/api/v1/embedding-search")
async def embedding_search(payload: SearchRequest):
    if not payload.docs or not payload.query:
        return {"matches": []}

    # Embed documents + query
    embeddings = embed_texts(payload.docs + [payload.query])
    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    # Compute similarity scores
    sims = [cosine_similarity(query_embedding, d) for d in doc_embeddings]

    # Get top 3 matches by similarity
    top3_idx = np.argsort(sims)[::-1][:3]
    matches = [payload.docs[i] for i in top3_idx]

    return {"matches": matches}
