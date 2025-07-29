import pytest
import numpy as np
from langchain_ollama import OllamaEmbeddings

def test_ollama_mxbai_embeddings():
    """Test ollama embeddings with mxbai-embed-large model."""
    
    # Initialize the embeddings model
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434"  # Default Ollama server URL
    )
    
    # Test texts
    texts = [
        "This is a test sentence for embeddings.",
        "Another completely different sentence.",
        "Something similar to the first sentence about testing embeddings."
    ]
    
    # Get embeddings for all texts
    embedded_texts = embeddings.embed_documents(texts)
    
    # Basic assertions
    assert len(embedded_texts) == 3, "Should have embeddings for all 3 texts"
    assert all(isinstance(emb, list) for emb in embedded_texts), "Each embedding should be a list"
    assert all(len(emb) == len(embedded_texts[0]) for emb in embedded_texts), "All embeddings should have same dimension"
    
    # Convert to numpy arrays for similarity calculation
    vectors = np.array(embedded_texts)
    
    # Calculate cosine similarities
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    # Test semantic similarities
    sim_1_2 = cosine_similarity(vectors[0], vectors[1])  # Should be lower (different meanings)
    sim_1_3 = cosine_similarity(vectors[0], vectors[2])  # Should be higher (similar meanings)
    
    print(f"\nSimilarity between dissimilar sentences: {sim_1_2:.4f}")
    print(f"Similarity between similar sentences: {sim_1_3:.4f}")
    
    # The similarity between similar sentences should be higher
    assert sim_1_3 > sim_1_2, "Similar sentences should have higher cosine similarity"

if __name__ == "__main__":
    test_ollama_mxbai_embeddings() 