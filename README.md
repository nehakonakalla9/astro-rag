# Astro RAG

A small Retrieval-Augmented Generation (RAG) pipeline, built from scratch, that answers questions about Vedic astrology house concepts using a custom document and the Gemini API.

## Why this exists

This is a learning project, not a polished product. I built it step by step to understand how RAG actually works under the hood: chunking, embeddings, vector search, and prompt construction, without hiding any of it behind a framework like LangChain. It's a stepping stone toward building agent-based systems with LangGraph next.

## How it works

**Setup stage (`ingest.py`), run once:**
1. Reads `astrology_docs.txt`, a document on Vedic astrology houses (compiled from Wikipedia and personal notes)
2. Splits it into 4 sections (House Definitions, Vedic House Names, House Groupings, Purusharthas & House Rounds)
3. Embeds each section into a vector using Gemini's `gemini-embedding-001` model
4. Stores the vectors and original text in a local Chroma vector database

**Question stage (`query.py`), run per question:**
1. Takes a question and embeds it with the same model
2. Searches the vector database for the top 2 closest chunks (cosine similarity)
3. Builds a prompt: instructions + retrieved chunks + the question
4. Sends it to Gemini (`gemini-3.5-flash`) and prints the answer

## Setup

1. Clone this repo and create a virtual environment:
  python -m venv venv
  venv\Scripts\activate
2. Install dependencies:
  pip install google-genai chromadb python-dotenv
3. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com).
4. Create a `.env` file in the project root with:
  GEMINI_API_KEY=your_actual_key_here
5. Run the ingestion script first (builds the vector database):
6. 
6. Edit the `question` variable in `query.py` to ask your own question, then run:
  python query.py



## Known limitations (and what I'd fix next)

- Chunking is done manually by section, not automatically. A 4-section document doesn't scale, a longer document would need fixed-size chunking with overlap, or automatic structure detection.
- The "House Definitions" chunk contains all 12 houses together, which is too broad. Splitting it into 12 separate chunks would improve retrieval precision.
- No evaluation yet, there's no test set checking whether retrieval actually returns the right chunk for a given question.
- Currently single-question, not a conversation, no chat history is retained.

## Stack

- Python
- Google Gemini API (`google-genai`)
- ChromaDB (vector database)
  
