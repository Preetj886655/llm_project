📄🤖 RAG-based PDF Chatbot using LangChain, LangGraph & HuggingFace

A smart Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions from a PDF document and get accurate, context-aware answers using modern LLM tooling.

This project combines LangChain, LangGraph, FAISS, and HuggingFace models to build an end-to-end conversational AI system. 🚀
🔗 **Live Demo:** [Click here to try the app](https://huggingface.co/spaces/Preetj2/LLM_Project)

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/Preetj2/LLM_Project)

✨ Features

✅ Upload and process PDF documents
✅ Chunking & semantic embeddings
✅ Fast similarity search using FAISS
✅ Tool-based retrieval with LangGraph
✅ LLM-powered conversational interface
✅ Modular & scalable architecture

🧠 How It Works

📂 Load PDF using PyPDFLoader

✂️ Split text into chunks

🔢 Generate embeddings using Sentence Transformers

🗂️ Store vectors in FAISS

🔍 Retrieve relevant chunks based on user query

🤝 Pass context to LLM using LangGraph tools

💬 Generate intelligent responses

🛠️ Tech Stack
🔹 Programming Language

Python 3.9+

🔹 Frameworks & Libraries

LangChain – LLM orchestration

LangGraph – Stateful conversational workflow

HuggingFace Transformers – LLM & embeddings

Sentence-Transformers – Semantic search

FAISS – Vector similarity search

PyPDF – PDF parsing

🔹 Models Used

DeepSeek-V3.2 (via HuggingFace Endpoint) – Text generation

all-MiniLM-L6-v2 – Embeddings model

🔹 Tools & Platforms

Google Colab / Jupyter Notebook

Git & GitHub

HuggingFace Hub

📦 Installation
pip install -U langchain langchain-core langchain-community langchain-huggingface langgraph
pip install pypdf faiss-cpu sentence-transformers

🔑 Setup

Set your HuggingFace token:

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"

▶️ Run the Project
python app.py


Then type your query:

user Type Message:- What is machine learning?

📂 Project Structure
.
├── app.py
├── data/
│   └── sample.pdf
├── requirements.txt
└── README.md

🎯 Use Cases

📚 Academic PDF assistant

🏢 Internal document chatbot

📑 Research paper Q&A system

🧑‍💻 AI-powered knowledge base

🚀 Future Improvements

Add Streamlit UI 🌐

Support multiple PDFs

Add conversation memory 🧠

Use reranking models for better accuracy

Deploy using Docker + AWS ☁️

🤝 Contributing

Contributions are welcome!
Feel free to fork this repo, raise issues, or submit pull requests. 🙌

📬 Contact

Preet Jaiswal
🎓 B.Tech Mechanical Engineering | Minor in Mathematical Computing
🤖 AI/ML & LLM Enthusiast
