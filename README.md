# RAG Chatbot (1M Token Dataset)

Scalable Retrieval-Augmented Generation using LangChain + Chroma

------------------------------------------------------------------------

## Overview

This project implements a scalable Retrieval-Augmented Generation (RAG)
chatbot capable of handling approximately **1,000,000 tokens of indexed
content**.

Instead of sending large datasets directly to the LLM, this
architecture:

1.  Indexes large corpora (\~1M tokens) into embeddings\
2.  Stores them in a Chroma vector database\
3.  Retrieves only relevant chunks per query\
4.  Injects retrieved context into the LLM for grounded answers

This enables scalable, memory-efficient question answering over large
datasets.

------------------------------------------------------------------------

## Architecture

Dataset (\~1M tokens)\
→ Recursive Chunking (400 / 80 overlap)\
→ OpenAI Embeddings\
→ Chroma Vector Store\
→ Adaptive Retriever (Similarity + MMR)\
→ LLM (gpt-4o-mini)\
→ Answer with Source Citations

------------------------------------------------------------------------

## Key Design Decisions

### 1. Designed for \~1 Million Tokens

-   The model does NOT receive 1M tokens at once.\
-   The vector database stores \~1M tokens.\
-   The retriever selects only relevant chunks (\~15 chunks per query).

This is true scalable RAG architecture.

------------------------------------------------------------------------

### 2. Chunking Strategy

``` python
chunk_size = 400
chunk_overlap = 80
```

**Why 400?** - Improves factual retrieval precision\
- Maintains semantic coherence\
- Ideal for fact-based queries (e.g., capitals, policies)

**Why 80 overlap?** - Prevents context loss at boundaries\
- Improves grounding\
- Reduces hallucinations

For a 1M token dataset this typically produces \~2,000--5,000 chunks.

------------------------------------------------------------------------

### 3. Adaptive Retrieval Strategy

Short queries (≤ 6 words):\
→ Similarity search

Longer queries:\
→ MMR (Max Marginal Relevance)

``` python
TOP_K = 15
FETCH_K = 60
lambda_mult = 0.5
```

Balances: - Relevance\
- Diversity\
- Reduced redundancy\
- Broader contextual coverage

------------------------------------------------------------------------

## Technology Stack

-   LangChain\
-   Chroma (Vector Database)\
-   OpenAI Embeddings (`text-embedding-3-small`)\
-   OpenAI Chat Model (`gpt-4o-mini`)\
-   Gradio (UI Layer)\
-   Google Colab (Runtime)

------------------------------------------------------------------------

## Features

-   Handles \~1M token datasets\
-   Supports PDF, TXT, MD, CSV\
-   Source citation (filename + page/row)\
-   Persistent vector storage\
-   Duplicate-safe indexing\
-   Debug retrieval mode

------------------------------------------------------------------------

## Performance Notes

For \~1M tokens:

-   Typical chunk count: 2k--5k\
-   Query latency: low (retrieves \~15 chunks)\
-   Memory footprint: stable with batch indexing

This architecture scales independently of the model's context window.

------------------------------------------------------------------------

## Why This Is True RAG

This project does not rely on increasing the model's context window.

Instead it:

-   Stores knowledge externally in a vector database\
-   Retrieves only relevant chunks\
-   Injects grounded context\
-   Minimizes hallucination risk\
-   Scales to multi-million token corpora

------------------------------------------------------------------------

## Future Improvements

-   Reranking layer (cross-encoder or LLM extractor)\
-   Hybrid search (BM25 + embeddings)\
-   FastAPI backend\
-   Cloud deployment (Cloud Run)\
-   Evaluation metrics & logging

------------------------------------------------------------------------

## Conclusion

This project demonstrates a production-style RAG architecture capable of
handling \~1 million tokens through:

-   Proper chunk sizing\
-   Stable vector indexing\
-   Adaptive retrieval\
-   Context-grounded generation

It provides a scalable foundation for large-document question answering
systems.
