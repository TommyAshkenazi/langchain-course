import os

from dotenv import load_dotenv

# from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")

    loader = TextLoader("mediumblog1.txt", encoding="UTF-8")
    document = loader.load()

    print("splitting document into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Number of chunks created: {len(texts)}")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ.get("PINECONE_INDEX_NAME")
    )
    print("Ingestion completed.")
