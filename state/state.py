from typing_extensions import TypedDict,Any

class ResearchPaperState(TypedDict):
    user_query: str
    paper: list
    artifacts: dict
    collection_name: str
    
    

class QAState(TypedDict):

    question:str

    collection_name:str

    paper:dict

    retrieved_context:list

    answer:str
    
    