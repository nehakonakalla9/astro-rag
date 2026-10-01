import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="astrology")

question = "What does the 4th house represent?"

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)
question_vector = result.embeddings[0].values
print(len(question_vector))

results = collection.query(
    query_embeddings=[question_vector],
    n_results=2
)

retrieved_chunks = results["documents"][0]
context = "\n\n---\n\n".join(retrieved_chunks)

prompt = f"""Answer the question using only the information below. If the answer isn't there, say you don't know.

Text:
{context}

Question: {question}
"""

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)

print(response.text)