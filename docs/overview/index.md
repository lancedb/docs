---
title: Overview of LanceDB, the Multimodal Vector Database?
hide:
  - toc
---
LanceDB is an open-source vector database for AI that's designed to store, manage, query and retrieve embeddings on large-scale multi-modal data. The core of LanceDB is written in Rust 🦀 and is built on top of [Lance](https://github.com/lancedb/lance), an open-source columnar data format designed for performant ML workloads and fast random access.

### What is "Multimodal" Database?

Most existing vector databases store and query just the embeddings and their metadata. The actual data is stored elsewhere, requiring you to manage their storage and versioning separately.

LanceDB supports storage of the *actual data itself*, alongside the embeddings and metadata. You can persist your images, videos, text documents, audio files and more in the Lance format, which provides automatic data versioning and blazing fast retrievals and filtering via LanceDB.

## LanceDB Cloud vs Self-Hosted

LanceDB **Cloud** is a SaaS (software-as-a-service) solution that runs serverless in the cloud, making the storage clearly separated from compute. It's designed to be cost-effective and highly scalable without breaking the bank. LanceDB Cloud is currently in private beta with general availability coming soon, but you can apply for early access with the private beta release by signing up below.

[Try out LanceDB Cloud (Public Beta) Now](https://cloud.lancedb.com){ .md-button .md-button--primary }

LanceDB **OSS** is an **open-source**, batteries-included embedded vector database that you can run on your own infrastructure. "Embedded" means that it runs *in-process*, making it incredibly simple to self-host your own AI retrieval workflows for RAG and more. No servers, no hassle.

!!! tip "Hosted LanceDB"
    If you want S3 cost-efficiency and local performance via a simple serverless API, checkout **LanceDB Cloud**. For private deployments, high performance at extreme scale, or if you have strict security requirements, talk to us about **LanceDB Enterprise**. [Learn more](https://docs.lancedb.com/)



