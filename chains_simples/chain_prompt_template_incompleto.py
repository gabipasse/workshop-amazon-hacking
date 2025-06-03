from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
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

if __name__ == "__main__":

    llm = ChatOllama(model="llama3")
    query = "What is {subject} in machine learning?"

    prompt_template = PromptTemplate.from_template(query)
    chain = prompt_template | llm

    # Lembrar que langhchain é praticamente um lego
    result = chain.invoke({"subject": "pinecone"})

    print("-------------------------RESPOSTA CRUA DA CHAIN--------------------------")
    print(result)

    print()
    print("-------------------------RESPOSTA TRATADA DA CHAIN-----------------------")
    print(result.content)
