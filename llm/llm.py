from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
)

llm1 = ChatGroq(
    model="meta-llama/llama-prompt-guard-2-86m",
    temperature=0.2,
)


from langchain_ollama import ChatOllama

gemma = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.3,
)

# from langchain_ollama import ChatOllama

# llm1 = ChatOllama(
#     model="gemma4:31b-cloud",
#     temperature=0.3,
# )