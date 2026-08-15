"""
verify_embeddings.py
---------------------
One-time sanity check for the local Sentence Transformer embedding setup.
Run this BEFORE your first real ingestion to confirm everything lines up.

Usage:
    python verify_embeddings.py

Checks:
  1. The embedding model loads successfully (downloads on first run only).
  2. It produces vectors of the expected dimension (384 for MiniLM-L6-v2).
  3. That dimension matches config.EMBEDDING_DIMENSION (used to create the
     Pinecone index) -- a mismatch here would cause every upsert to fail.
  4. A real round-trip: embed -> upsert one throwaway vector to Pinecone ->
     fetch it back -> delete it. Confirms write+read actually works, not
     just that the model loads.
"""

from config import EMBEDDING_DIMENSION, get_embeddings, get_or_create_index

print("1. Loading local embedding model (first run downloads ~90MB)...")
embeddings = get_embeddings()
print("   OK\n")

print("2. Embedding a test query...")
vector = embeddings.embed_query("What was total revenue for fiscal year 2024?")
print(f"   Produced a {len(vector)}-dimensional vector\n")

print(f"3. Checking against config.EMBEDDING_DIMENSION ({EMBEDDING_DIMENSION})...")
if len(vector) != EMBEDDING_DIMENSION:
    print(
        f"   ❌ MISMATCH: model produced {len(vector)} dims but config says "
        f"{EMBEDDING_DIMENSION}. Fix EMBEDDING_DIMENSION in config.py to match "
        f"your actual embedding model, then delete and recreate your Pinecone "
        f"index (dimension can't be changed on an existing index)."
    )
    raise SystemExit(1)
print("   OK, dimensions match\n")

print("4. Round-trip test against Pinecone (upsert -> fetch -> delete)...")
index = get_or_create_index()
test_id = "__verify_embeddings_test__"
index.upsert(vectors=[(test_id, vector, {"test": True})])
result = index.fetch(ids=[test_id])
if test_id not in (result.vectors or {}):
    print("   ❌ Upserted a vector but couldn't fetch it back -- check your Pinecone setup.")
    raise SystemExit(1)
index.delete(ids=[test_id])
print("   OK, wrote and read back a real vector successfully\n")

print("✅ All checks passed. Your embedding pipeline is correctly wired to Pinecone.")
