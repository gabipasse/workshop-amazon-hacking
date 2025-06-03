from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Link download ollama (default: cpu)
# ollama.com/download

# Link pinecone
# pinecone.io

# Link langsmith
# smith.langchain.com

# Nos imports e objetos, é importante sair dando "." em tudo para conseguir perceber
# todas as possibilidades existentes. Checar atributos, métodos, módulos, etc.


# Desenho preenchimento Field pela LLM.
# Output parsers são importantes para obrigar chain a retornar objeto especifico.
# Feature importante para popular frontend.
class Summary(BaseModel):
    summary: str = Field(description="summary")
    facts: List[str] = Field(description="interesting facts about them")

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "facts": self.facts}


# Normalmente evitem misturar programação orientada a objeto e programação funcional
def format_answer_for_streamlit(response: Summary) -> str:
    response_dict = response.to_dict()
    facts = response_dict.get("facts", [])

    formatted_facts = "\n".join(f"* {fact}" for fact in facts)

    return f"* {response_dict.get('summary')}\n{formatted_facts}"


if __name__ == "__main__":

    llm = ChatOllama(model="llama3")
    summary_parser = PydanticOutputParser(pydantic_object=Summary)

    query = """Given the {animal}. I want you to create:
    1. A short summary
    2. Interesting facts about it
    \n{format_instructions}"""

    summary_prompt_template = PromptTemplate(
        input_variables=["animal"],
        template=query,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    chain = summary_prompt_template | llm | summary_parser

    # Lembrar que langhchain é praticamente um lego
    result = chain.invoke({"animal": "tiger"})
    formatted_result = format_answer_for_streamlit(result)

    print("-------------------------RESPOSTA CRUA DA CHAIN--------------------------")
    print(result)

    print()
    print("-------------------------RESPOSTA TRATADA DA CHAIN-----------------------")
    print(formatted_result)
