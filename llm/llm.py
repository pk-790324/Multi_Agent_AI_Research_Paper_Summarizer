from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="deepseek-v4.1-flash:cloud",
    temperature=0.3,
)

llm1 = ChatOllama(
    model="deepseek-v4.1-flash:cloud",
    temperature=0.3,
)


gemma = ChatOllama(
    model="deepseek-v4.1-flash:cloud",
    temperature=0.3,
)


# from langchain_ollama import ChatOllama

# llm1 = ChatOllama(
#     model="gemma4:31b-cloud",
#     temperature=0.3,
# )