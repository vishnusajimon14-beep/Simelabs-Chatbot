import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# load_dotenv()

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
OPENROUTER_API_KEY=st.secrets["OPENROUTER_API_KEY"]
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat-v3"
)

def ask_question(question):
    retrieved_docs = retriever.invoke(question)

    context="\n".join([doc.page_content for doc in retrieved_docs])
    sources=list(set(doc.metadata["source"]
                     for doc in retrieved_docs))
    
    prompt=f"""
    You are simelabs Ai assistant
Rules:
1. If the user is greeting you, thanking you, or engaging in casual conversation,
   respond naturally and politely.

2. If the user asks about SimeLabs, answer using ONLY the provided context.

3. If the answer is not available in the context, say:
   "I could not find that information on the SimeLabs website."
 Context:{context}
Question:{question}
"""
    response=llm.invoke(prompt)
    return response.content,sources



st.set_page_config(
    page_title="Simelabs Chatbot assistant",
    page_icon="simelabs-logo.png",
    layout="wide"
)
st.logo("simelabs-logo.png")
st.markdown(
    "<h1 style='text-align:center;'>SimeLabs AI Assistant</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Ask anything about SimeLabs services, careers, industries and company information</p>",
    unsafe_allow_html=True
)

# st.divider()

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
question=st.chat_input("Ask a question")
if question:
    st.session_state.messages.append({
        "role":"user",
        "content":question
    })
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching simelabs knowledge base.."):
            answer,sources=ask_question(question)
            st.markdown(answer)
            # st.markdown(answer)
            st.markdown("##sources")
            for source in sources:
                st.markdown(f"-{source}")
    st.session_state.messages.append({
        "role":"assistant",
        "content":answer
    })

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center'>
        © 2026 SimeLabs Technologies Pvt Ltd<br>
        Powered by RAG + Pinecone + DeepSeek
    </div>
    """,
    unsafe_allow_html=True
)
