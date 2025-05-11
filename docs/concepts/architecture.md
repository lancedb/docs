# Architecture

```text
┌────────────────────────────────────────────────────┐
│ Data Ingestion                                     │
├────────────────────────────────────────────────────┤
│ ┌────────────┐   ┌──────────────┐   ┌────────────┐ │
│ │ CSV / JSON │ → │ Embeddings   │ → │ Lance Table│ │
│ └────────────┘   └─────┬────────┘   └────┬───────┘ │
│                        │                 │         │
│                        ↓                 ↓         │
│        HuggingFace / OpenAI          Local or      │
│        SentenceTransformers          S3/Blob       │
│            (128D / 768D)             Storage       │
└────────────────────────────────────────────────────┘
Raw data is stored in csv or json. You can then embed 
it and insert it into a Lance table. Both the data and
the table can be stored in the same S3 Blob storage.
                           │
                           ↓
┌────────────────────────────────────────────────────┐
│ LanceDB Engine                                     │
│ (Local or Remote FS)                               │
└────────────────────────────────────────────────────┘
The LanceDB engine is the core component that manages
data storage and retrieval. You can run it locally on 
your machine or on a remote filesystem.
                           │
                           ↓
┌────────────────────────────────────────────────────┐
│ Index Management                                   │
├────────────────────────────────────────────────────┤
│ - IVF_PQ / HNSW / Flat Index                       │
│ - Metadata Indices (for filters)                   │
└────────────────────────────────────────────────────┘
Index management handles vector search optimization.
IVF_PQ and HNSW are specialized vector indices for
fast similarity search. Metadata indices enable
efficient filtering of search results based on
non-vector attributes.
                           │
                           ↓
┌────────────────────────────────────────────────────┐
│ Query Interface                                    │
├────────────────────────────────────────────────────┤
│ - Python SDK (Jupyter)                             │
│ - REST API / FastAPI                               │
│ - CLI / LangChain Agent                            │
└────────────────────────────────────────────────────┘
Multiple interfaces for interacting with LanceDB:
- Python SDK for direct integration in Jupyter notebooks
- REST API for web applications and microservices
- CLI and LangChain integration for automation
                           │
                           ↓
┌────────────────────────────────────────────────────┐
│ Application Layer (Frontend)                       │
├────────────────────────────────────────────────────┤
│ - Semantic Search App                              │
│ - Chatbot / RAG Pipeline                           │
│ - Analytics / Exploration                          │
└────────────────────────────────────────────────────┘
End-user applications built on top of LanceDB:
- Semantic search applications for finding similar content
- Chatbots and RAG pipelines for AI-powered interactions
- Analytics tools for exploring and visualizing data
```