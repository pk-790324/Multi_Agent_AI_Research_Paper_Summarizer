from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=2048,
)

llm1 = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=2048,
)


from langchain_ollama import ChatOllama

# llm = ChatOllama(
#     model="gemma4:31b-cloud",
#     temperature=0.3,
# )

# from langchain_ollama import ChatOllama

# llm1 = ChatOllama(
#     model="gemma4:31b-cloud",
#     temperature=0.3,
# )