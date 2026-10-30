from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=2048,
)