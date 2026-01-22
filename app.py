import os
import tempfile
import streamlit as st

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool

from typing import TypedDict, Annotated

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 Upload PDF & Chat (RAG)")
st.write("Upload a PDF and ask questions from its content")

# -------------------------------
# HF TOKEN (Secrets)
# -------------------------------


# -------------------------------
# LOAD LLM (Cached)
# -------------------------------
@st.cache_resource
def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3.2",
        task="text-generation"
    )
    return ChatHuggingFace(llm=llm)

model = load_llm()

# -------------------------------
# PDF UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

# -------------------------------
# BUILD VECTOR STORE FROM PDF
# -------------------------------
@st.cache_resource(show_spinner="Processing PDF...")
def build_vectorstore(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=30
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# -------------------------------
# LANGGRAPH STATE
# -------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------------------
# UI LOGIC
# -------------------------------
if uploaded_file:
    vector_store = build_vectorstore(uploaded_file.read())
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    @tool
    def rag_tool(query: str):
        """Retrieve relevant info from uploaded PDF"""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    tools = [rag_tool]
    model_with_tools = model.bind_tools(tools)

    tool_node = ToolNode(tools)

    def chat_node(state: ChatState):
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    chatbot = graph.compile()

    # -------------------------------
    # CHAT UI
    # -------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.chat_input("Ask a question from the PDF...")

    if user_query:
        st.session_state.chat_history.append(("user", user_query))

        result = chatbot.invoke({
            "messages": [HumanMessage(content=user_query)]
        })

        answer = result["messages"][-1].content
        st.session_state.chat_history.append(("bot", answer))

    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

else:
    st.info("⬆️ Upload a PDF to start chatting")
