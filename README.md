# 🩺 MEDIBOT – AI Medical Chatbot with RAG

**MEDIBOT** is an AI-powered medical chatbot built using Retrieval-Augmented Generation (RAG). It retrieves relevant information from medical documents and uses a Large Language Model (LLM) to generate context-based responses to users' questions.

The application provides an interactive chat interface using Streamlit and uses Groq to generate responses.

## ✨ Features

- **PDF-Based Knowledge Retrieval:** Retrieves relevant information from medical PDF documents.
- **RAG Architecture:** Combines document retrieval with LLM-based response generation.
- **Hugging Face Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` to convert text into embeddings.
- **FAISS Vector Database:** Stores and searches document embeddings efficiently.
- **Groq LLM Integration:** Uses `openai/gpt-oss-20b` to generate responses.
- **Interactive Chat Interface:** Built with Streamlit.
- **Source Documents:** Displays retrieved text and its source page.
- **Context-Based Responses:** Instructs the chatbot not to invent answers when the retrieved context is insufficient.

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Hugging Face
- Sentence Transformers
- FAISS
- Groq API
- PyPDF

## 🏗️ Architecture

1. **Document Loading:** Medical PDF documents are loaded from the data directory.
2. **Text Splitting:** Documents are divided into smaller chunks.
3. **Embedding Generation:** Hugging Face embeddings convert text chunks into numerical vectors.
4. **Vector Storage:** FAISS stores the generated embeddings.
5. **Similarity Search:** Relevant chunks are retrieved based on the user's question.
6. **LLM Response Generation:** Groq's GPT-OSS-20B generates an answer using the retrieved context.
7. **Response Display:** Streamlit displays the answer and its source documents.