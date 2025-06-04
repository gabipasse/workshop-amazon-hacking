import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader(
        r"full_path",
        encoding="UTF-8",
    )
    document_generator = loader.load()

    print()
    print("Splitting...")
    # You can increase chunk_size or use chunk_overlap. Might also try modifying separator to "."
    text_chunk_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    text_chunks = text_chunk_splitter.split_documents(document_generator)
    print(f"created {len(text_chunks)} chunks")

    print()
    print("Ingesting into vector database...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    PineconeVectorStore.from_documents(
        text_chunks, embeddings, index_name=os.environ["INDEX_NAME"]
    )

    print("Finish!")
