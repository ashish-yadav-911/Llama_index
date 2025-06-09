from openai import OpenAI
from dotenv import load_dotenv
import os
from pinecone import Pinecone

# This script uploads text chunks to a Pinecone index using OpenAI embeddings.
load_dotenv()
print("Loaded environment variables.")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("Initialized OpenAI client.")

def embed_text(text):
    print(f"Embedding text: {text[:60]}...")
    embedding = client.embeddings.create(
        model="text-embedding-3-large",  # or any embedding model
        input=[text]
    ).data[0].embedding
    print(f"Embedding generated. Length: {len(embedding)}")
    return embedding


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print("Initialized Pinecone client.")
index = pc.Index(host="https://iffort-xtmbcry.svc.aped-4627-b74a.pinecone.io")
print("Connected to Pinecone index.")
records = []
chunks = [
    [
    {
        "_id": "iffort-1",
        "chunk_text": "Innovative AI Solutions: Our intelligent AI solutions give forward-thinking companies a friendly boost, helping them work better and stand out. We distinguish between signal and noise in the realm of AI.",
        "category": "ai-solutions"
    },
    {
        "_id": "iffort-2",
        "chunk_text": "Conversational AI Agents: Unlock the potential of our Conversational AI agents, designed to automate both chat and voice interactions. Our human-like AI agents handle customer queries 24/7, boosting efficiency and driving conversions. Supports multi-language conversations, integrates with booking systems, and automates workflows.",
        "category": "conversational-ai"
    },
    {
        "_id": "iffort-3",
        "chunk_text": "Voice AI Agent for F&B Industry: Multi-lingual voice AI agent for 24/7 restaurant bookings. Handles reservations in multiple languages for a smooth customer experience.",
        "category": "voice-ai"
    },
    {
        "_id": "iffort-4",
        "chunk_text": "AI Training & Workshops: Specialized workshops to help leaders, managers, and executives navigate AI confidently. Focused on strategies, productivity, and innovation. Key Takeaways include developing AI strategies, cultivating an AI-first culture, and leading AI-driven change.",
        "category": "training"
    },
    {
        "_id": "iffort-5",
        "chunk_text": "Custom AI & ML Development: We develop custom AI across industries. Examples include ChefGPT (personalized recipe generator), Digital Friend (chatbot for parents of kids with Down Syndrome), and Mental Health Chatbot for kids with ADHD.",
        "category": "custom-ai"
    },
    {
        "_id": "iffort-6",
        "chunk_text": "Mirai for Real Estate Industry: AI assistant to qualify leads, assess eligibility, and recommend properties. Saves time and improves client engagement for more transactions.",
        "category": "real-estate"
    },
    {
        "_id": "iffort-7",
        "chunk_text": "Why Work with Us: Members of Forbes Agency Council, founding member of AI Marketers Guild, and recognized partners for platforms like Vapi and KOGO. Experts in seamless AI integration.",
        "category": "about-us"
    },
    {
        "_id": "iffort-8",
        "chunk_text": "Meet the Humans Behind AI: Iffort.ai is a business unit of Iffort, a tech and digital marketing company with 14+ years of experience. Team includes Daksh Sharma, Sunny Jindal, Ashish Upadhyay, and others.",
        "category": "team"
    },
    {
        "_id": "iffort-9",
        "chunk_text": "Company Overview: Iffort is a tech and marketing firm founded in 2010. Offices in Delhi, Dubai, and Canada. 100+ projects, 200 brands served, $10mn+ performance media budget.",
        "category": "company-overview"
    },
    {
        "_id": "iffort-10",
        "chunk_text": "Office Locations: Noida (India), Sharjah (UAE), Oshawa (Canada). Contact at business@iffort.com.",
        "category": "locations"
    }
]

]

print(f"Total chunks: {len(chunks[0])}")
for chunk in chunks[0]:
    print(f"Processing chunk ID: {chunk['_id']}")
    embedding = embed_text(chunk["chunk_text"])
    record = {
        "id": chunk["_id"],
        "values": embedding,
        "metadata": {
            "chunk_text": chunk["chunk_text"],
            "category": chunk["category"]
        }
    }
    print(f"Record ready for upsert: {record['id']}")
    records.append(record)

print(f"Upserting {len(records)} records to Pinecone...")
index.upsert(vectors=records, namespace="iffort_details")
print("Upsert complete.")
