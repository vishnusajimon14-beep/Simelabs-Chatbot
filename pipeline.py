import pandas as pd
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

df=pd.read_csv("simelabs_knowledge_base.csv")
documents=[]
for _,row in df.iterrows():
    url = row["url"]

    if "/services/" in url:
        page_type = "service"

    elif "/careers/" in url:
        page_type = "career"

    elif "/industries/" in url:
        page_type = "industry"

    elif "/company/" in url:
        page_type = "company"

    else:
        page_type = "general"
    
    content = f"""
    URL: {url}

    {row['content']}
    """

    doc = Document(
        page_content=content,
        metadata={
            "source": url,
            "type": page_type
        }
    )

    documents.append(doc)
print(f"documents:{len(documents)}")
# print(documents[0])

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
print(f"Total Chunks: {len(chunks)}")

from langchain_huggingface import HuggingFaceEmbeddings
embedding=HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
# vector = embedding.embed_query("Artificial Intelligence")
# print(len(vector))

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
pc = Pinecone(
    api_key=PINECONE_API_KEY
)
index = pc.Index(PINECONE_INDEX_NAME)
index.delete(delete_all=True)
PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        index_name=PINECONE_INDEX_NAME
    )
print("upload complete")