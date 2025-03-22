# QA CLI Tool

A question-answering tool that leverages language models, retrieval-augmented generation (RAG), and internet search capabilities to provide accurate responses.

## Configuration

You must specify API_KEY in .env file to use deepresearch.

## Features

- **LightResearch**: Basic Q&A with configurable options for:
  - Internet search integration
  - RAG (Retrieval Augmented Generation)
  - Advanced RAG

- **DeepResearch**: More comprehensive analysis using advanced models (e.g., Gemini 2.0)

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) installed locally for embedding and chat models
- API keys configured in `.env` file

## Installation

This project uses `uv` for dependency management.


## Usage

Run the CLI tool from the project root:

```bash
uv run main.py
```

The CLI will:
1. Initialize the Chroma vector database
2. Prompt for a question
3. Ask if you want to use DeepResearch or LightResearch
4. Configure additional options based on your selection

### LightResearch Options

- Internet search: Uses DuckDuckGo to retrieve information
- RAG: Uses local document storage for context retrieval
- Advanced RAG: Implements more sophisticated retrieval strategies

### DeepResearch

Leverages the model specified in the `.env` file for more comprehensive analysis.

You can edit topics to analyze via human feedback.

## Data

Place any documents you want to include in your knowledge base in the `data/` directory. The system will index these documents for RAG functionality.
