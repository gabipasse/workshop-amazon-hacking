from langchain_community.llms import LlamaCpp
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

# TODO
# bash windows: huggingface-cli download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF --include "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf" --local-dir ./models --local-dir-use-symlinks False


if __name__ == "__main__":

    model_path = r"models\Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

    # Pode modificar parametros como temperatura, top_p e top_k
    llm = LlamaCpp(
        model_path=model_path, temperature=0.7, max_tokens=512, n_ctx=2048, verbose=True
    )
    query = (
        "[INST] <<SYS>>\n"
        "You are a helpful, polite, and knowledgeable assistant.\n"
        "<</SYS>>\n\n"
        "Explain {subject} like I'm 5 years old. [/INST]"
    )

    prompt_template = PromptTemplate.from_template(query)
    chain = prompt_template | llm

    # Lembrar que langhchain é praticamente um lego
    result = chain.invoke({"subject": "quantum physics"})

    print("-------------------------RESPOSTA CRUA DA CHAIN--------------------------")
    print(result)
