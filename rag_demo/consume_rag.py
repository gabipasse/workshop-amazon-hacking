import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

# Link ollama

# Link pinecone

if __name__ == "__main__":
    print("Retrieving...")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3")

    query = "What is pinecone in machine learning?"
    vector_store = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    # The value for the key 'context' is, by default, populated with the documents returned from the retriever
    retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    # The retriever uses, as default, the value associated to the key 'input' to define the query for retriever
    result = retrieval_chain.invoke(input={"input": query})

    print(type(result))
    print()
    print(result.get("answer"))
