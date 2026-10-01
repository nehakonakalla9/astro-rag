import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

with open("astrology_docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

titles = ["House Definitions", "Vedic House Names", "House Groupings", "Purusharthas & House Rounds"]

chunks = text.split("SECTION:")
chunks = chunks[1:]

clean_chunks = []

for i in range(4):
    title = titles[i]
    content = chunks[i]

    content = content.split("\n", 1)[1]
    content = content.strip()

    combined = "Section: " + title + "\n\n" + content
    clean_chunks.append(combined)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.create_collection(name="astrology")

for i in range(4):
    chunk_text = clean_chunks[i]

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk_text
    )
    vector = result.embeddings[0].values

    collection.add(
        ids=["chunk_" + str(i)],
        embeddings=[vector],
        documents=[chunk_text]
    )

print("Done, stored", collection.count(), "chunks")