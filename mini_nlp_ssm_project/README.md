# Mini NLP SSM Project

A small **Semantic Similarity Module (SSM)** using TensorFlow + Universal Sentence Encoder.

This project demonstrates a lightweight NLP module that:
- Computes embeddings for sentences
- Returns top-k most similar sentences to a query
- Can be served via a FastAPI API

## Features
- GPU-enabled with TensorFlow
- FastAPI endpoint: `/similar?sentence=...&top_k=3`

## Installation
```bash
cd mini_nlp_ssm_project