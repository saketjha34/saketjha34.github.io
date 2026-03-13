---
layout: post
title: Semantic Search with FAISS From Theory to Production
date: 2025-09-23 17:00:00
description: Semantic search has revolutionized how we find and retrieve information. Unlike traditional keyword-based search, semantic search understands the meaning and context behind queries, enabling more intelligent and relevant results. In this comprehensive guide, we'll explore how to implement production ready semantic search systems using FAISS (Facebook AI Similarity Search) and sentence transformers.
tags: [semantic-search, vector-store]
categories: NLP
# thumbnail: assets/img/9.jpg
pinned: true
---

## Understanding Semantic Search

Traditional search relies on exact keyword matches:
- Query: "fast car"
- Matches: Documents containing exactly "fast" and "car"

Semantic search understands meaning:
- Query: "fast car" 
- Matches: Documents about "speedy vehicles", "quick automobiles", "rapid transportation"

## Why FAISS?

FAISS is a library for efficient similarity search and clustering of dense vectors, offering:

- **Speed**: Optimized for billion-scale vector search
- **Memory Efficiency**: Various indexing strategies for different use cases
- **GPU Support**: Accelerated search on CUDA-enabled devices
- **Flexibility**: Multiple distance metrics and index types

## Architecture Overview

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class SemanticSearchEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def build_index(self, documents: List[str]):
        """Build FAISS index from documents"""
        self.documents = documents
        self.embeddings = self.model.encode(documents)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append((self.documents[idx], float(score)))
        
        return results
```

## Advanced Indexing Strategies

### 1. IVF (Inverted File) Index

For large-scale datasets, use IVF indexing:

```python
def create_ivf_index(embeddings: np.ndarray, nlist: int = 100):
    """Create IVF index for faster search on large datasets"""
    dimension = embeddings.shape[1]
    
    # Create quantizer
    quantizer = faiss.IndexFlatIP(dimension)
    
    # Create IVF index
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # Train the index
    index.train(embeddings.astype('float32'))
    index.add(embeddings.astype('float32'))
    
    # Set search parameters
    index.nprobe = 10  # Number of clusters to search
    
    return index
```

### 2. HNSW (Hierarchical Navigable Small World)

For ultra-fast approximate search:

```python
def create_hnsw_index(embeddings: np.ndarray, M: int = 16):
    """Create HNSW index for fast approximate search"""
    dimension = embeddings.shape[1]
    
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = 200  # Construction parameter
    index.hnsw.efSearch = 50         # Search parameter
    
    index.add(embeddings.astype('float32'))
    return index
```

## Production-Ready Implementation

### Complete Search System

```python
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class SearchResult:
    document: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

class ProductionSemanticSearch:
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        cache_dir: str = "./cache"
    ):
        self.model = SentenceTransformer(model_name)
        self.index_type = index_type
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.index = None
        self.documents = []
        self.metadata = []
        
        self.logger = logging.getLogger(__name__)
    
    def build_index(
        self, 
        documents: List[str], 
        metadata: Optional[List[Dict]] = None,
        save_path: Optional[str] = None
    ):
        """Build and optionally save the search index"""
        self.documents = documents
        self.metadata = metadata or [{}] * len(documents)
        
        # Generate embeddings
        self.logger.info(f"Encoding {len(documents)} documents...")
        embeddings = self.model.encode(
            documents, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # Create appropriate index
        dimension = embeddings.shape[1]
        
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)
        elif self.index_type == "ivf":
            nlist = min(int(np.sqrt(len(documents))), 1000)
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.train(embeddings.astype('float32'))
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 16)
        
        # Normalize and add embeddings
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Save if requested
        if save_path:
            self.save_index(save_path)
        
        self.logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def search(
        self, 
        query: str, 
        k: int = 10,
        filter_func: Optional[callable] = None
    ) -> List[SearchResult]:
        """Search with optional filtering"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        search_k = k * 2 if filter_func else k  # Get more results for filtering
        scores, indices = self.index.search(
            query_embedding.astype('float32'), 
            search_k
        )
        
        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
                
            metadata = self.metadata[idx]
            
            # Apply filter if provided
            if filter_func and not filter_func(metadata):
                continue
            
            results.append(SearchResult(
                document=self.documents[idx],
                score=float(score),
                metadata=metadata
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def save_index(self, path: str):
        """Save index and metadata to disk"""
        save_path = Path(path)
        save_path.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save documents and metadata
        with open(save_path / "documents.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "model_name": self.model.get_sentence_embedding_dimension()
            }, f)
        
        self.logger.info(f"Index saved to {save_path}")
    
    def load_index(self, path: str):
        """Load index and metadata from disk"""
        load_path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "index.faiss"))
        
        # Load documents and metadata
        with open(load_path / "documents.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
        
        self.logger.info(f"Index loaded from {load_path}")
```

## Performance Optimization

### Batch Processing

```python
def batch_encode_documents(
    model: SentenceTransformer,
    documents: List[str],
    batch_size: int = 32
) -> np.ndarray:
    """Efficiently encode large document collections"""
    embeddings = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)
```

### GPU Acceleration

```python
def create_gpu_index(embeddings: np.ndarray, gpu_id: int = 0):
    """Create GPU-accelerated FAISS index"""
    dimension = embeddings.shape[1]
    
    # Create CPU index first
    cpu_index = faiss.IndexFlatIP(dimension)
    
    # Move to GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    
    # Add embeddings
    gpu_index.add(embeddings.astype('float32'))
    
    return gpu_index
```

## Real-World Example: Document Search API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Semantic Search API")

# Initialize search engine
search_engine = ProductionSemanticSearch(
    model_name="all-MiniLM-L6-v2",
    index_type="hnsw"
)

class SearchRequest(BaseModel):
    query: str
    k: int = 10
    filters: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_time: float

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents using semantic similarity"""
    import time
    start_time = time.time()
    
    try:
        # Define filter function if filters provided
        filter_func = None
        if request.filters:
            def filter_func(metadata):
                for key, value in request.filters.items():
                    if metadata.get(key) != value:
                        return False
                return True
        
        # Perform search
        results = search_engine.search(
            query=request.query,
            k=request.k,
            filter_func=filter_func
        )
        
        total_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            total_time=total_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/build")
async def build_index(documents: List[str], metadata: Optional[List[Dict]] = None):
    """Build search index from documents"""
    try:
        search_engine.build_index(documents, metadata)
        return {"message": f"Index built with {len(documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Evaluation Metrics

### Measuring Search Quality

```python
def evaluate_search_quality(
    search_engine: ProductionSemanticSearch,
    test_queries: List[str],
    ground_truth: List[List[str]],
    k: int = 10
) -> Dict[str, float]:
    """Evaluate search quality using standard metrics"""
    
    def calculate_precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant))
        return relevant_retrieved / min(k, len(retrieved_k)) if retrieved_k else 0
    
    def calculate_recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant))
        return relevant_retrieved / len(relevant) if relevant else 0
    
    precisions = []
    recalls = []
    
    for query, relevant_docs in zip(test_queries, ground_truth):
        results = search_engine.search(query, k=k)
        retrieved_docs = [r.document for r in results]
        
        precision = calculate_precision_at_k(retrieved_docs, relevant_docs, k)
        recall = calculate_recall_at_k(retrieved_docs, relevant_docs, k)
        
        precisions.append(precision)
        recalls.append(recall)
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return {
        "precision@k": avg_precision,
        "recall@k": avg_recall,
        "f1@k": f1_score
    }
```

## Mathematical Foundation

The similarity between query and document embeddings is calculated using cosine similarity:

$$\text{similarity}(q, d) = \frac{q \cdot d}{||q|| \cdot ||d||}$$

For normalized vectors, this simplifies to the dot product:

$$\text{similarity}(q, d) = q \cdot d$$

The search complexity for different index types:

- **Flat Index**: $O(n \cdot d)$ where $n$ is number of documents, $d$ is dimension
- **IVF Index**: $O(\sqrt{n} \cdot d)$ with proper clustering
- **HNSW Index**: $O(\log n \cdot d)$ for approximate search

## Deployment Considerations

### Docker Configuration

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Monitoring and Logging

```python
import structlog
from prometheus_client import Counter, Histogram

# Metrics
SEARCH_REQUESTS = Counter('search_requests_total', 'Total search requests')
SEARCH_DURATION = Histogram('search_duration_seconds', 'Search request duration')

logger = structlog.get_logger()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    logger.info(
        "request_completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        duration=duration
    )
    
    SEARCH_REQUESTS.inc()
    SEARCH_DURATION.observe(duration)
    
    return response
```

## Conclusion

Semantic search with FAISS enables powerful, scalable similarity search systems. Key takeaways:

1. **Choose the Right Index**: Flat for accuracy, IVF/HNSW for scale
2. **Optimize Embeddings**: Use appropriate models and normalization
3. **Monitor Performance**: Track both speed and quality metrics
4. **Plan for Scale**: Consider GPU acceleration and distributed systems
5. **Evaluate Continuously**: Use proper metrics to measure search quality

The combination of modern transformer models and efficient vector search makes semantic search accessible for production applications, opening new possibilities for intelligent information retrieval.