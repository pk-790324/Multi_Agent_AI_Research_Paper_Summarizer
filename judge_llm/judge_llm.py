
from deepeval.models import DeepEvalBaseLLM

from llm.llm import llm1
llm=llm1


class CustomLLM(DeepEvalBaseLLM):

    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:

        response = self.model.invoke(prompt)

        return response.content

    async def a_generate(self, prompt: str) -> str:

        response = await self.model.ainvoke(prompt)

        return response.content

    def get_model_name(self):

        return "My-Project-LLM"


# Initialize your existing LLM
judge_llm = CustomLLM(model=llm)
