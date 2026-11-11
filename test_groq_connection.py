import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY =os.getenv("GROQ_API_KEY")

llm=ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5,
    max_tokens=150,
    api_key=GROQ_API_KEY
)

response=llm.invoke(
    "Explain artificial intelligence in two sentences."
)
print("\nGroq Connection Successful!")
print(response.content)
