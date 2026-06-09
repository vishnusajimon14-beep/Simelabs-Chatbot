import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
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

contextualize_q_prompt=ChatPromptTemplate.from_messages(
    [
        (
             "system",
            """
Given a chat history and latest user question,
rewrite the question into a standalone question.

Do NOT answer the question.

Only rewrite it if needed.
"""
        ),MessagesPlaceholder("chat_history"),("human","{input}")
    ]
)
history_aware_retriver=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

qa_prompt=ChatPromptTemplate.from_messages([
    ("system","""
    you are simelab ai assistant ,use the context and history to answer.
     Rules:
     1. use only information from the context.
     2.Do not use outside knowledge.
     3.if answer is not in context,say :'i could not find that information on the simelabs website'
     context:{context}
     """),MessagesPlaceholder("chat_history"),("human","{input}")
])

def ask_question(question,chat_history):
    print("Question",question)
    retrieved_docs = history_aware_retriver.invoke(
         {
            "input": question,
            "chat_history": chat_history
        }
    )
    print("\nRETRIEVED DOCS")
    for i, doc in enumerate(retrieved_docs):
        print(f"\nChunk {i+1}")
        print(doc.page_content[:500])

    context="\n".join([doc.page_content for doc in retrieved_docs])
    sources=list(set(doc.metadata.get("source","Unknown")
                     for doc in retrieved_docs))
    
    chain=qa_prompt|llm
    response=chain.invoke({
        "input":question,
        "chat_history":chat_history,
        "context":context
    })
    
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
    recent_messages=st.session_state.messages[-8:-1]
    chat_history=[]
    for msg in recent_messages:
        if msg["role"]=="user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))
    with st.chat_message("assistant"):
        with st.spinner("Searching simelabs knowledge base.."):
            answer,sources=ask_question(question,chat_history)
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
