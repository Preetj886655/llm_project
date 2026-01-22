import os
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
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("📄 RAG-based PDF Chatbot")
st.write("Ask questions from your uploaded PDF using DeepSeek + LangGraph")

# -------------------------------
# ENV VARIABLE (HF TOKEN)
# -------------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# -------------------------------
# LOAD LLM
# -------------------------------
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# -------------------------------
# LOAD & PROCESS PDF (cached)
# -------------------------------
@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader("22MEB0A46_TFTTHINK.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=30
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vector_store = load_vectorstore()
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# -------------------------------
# LANGGRAPH STATE
# -------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------------------
# RAG TOOL
# -------------------------------
@tool
def rag_tool(query: str):
    """Retrieve relevant info from PDF"""
    docs = retriever.invoke(query)
    context = [doc.page_content for doc in docs]
    return "\n\n".join(context)

tools = [rag_tool]
model_with_tools = model.bind_tools(tools)

tool_node = ToolNode(tools)

# -------------------------------
# CHAT NODE
# -------------------------------
def chat_node(state: ChatState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# -------------------------------
# BUILD GRAPH
# -------------------------------
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

user_query = st.chat_input("Ask something from the PDF...")

if user_query:
    st.session_state.chat_history.append(("user", user_query))

    result = chatbot.invoke({
        "messages": [HumanMessage(content=user_query)]
    })

    answer = result["messages"][-1].content
    st.session_state.chat_history.append(("bot", answer))

# -------------------------------
# DISPLAY CHAT
# -------------------------------
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
