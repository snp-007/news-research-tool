# 📰 News Research Tool using Groq, RAG & FAISS

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge\&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Database-green?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-RAG-success?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

<a href="https://news-research-tool-snp.streamlit.app/">Try the App</a>


# 📌 Project Overview

News Research Tool is a Retrieval-Augmented Generation (RAG) application that enables users to analyze and query multiple news articles using natural language.

Instead of manually reading articles or copying large amounts of text into ChatGPT, users simply provide article URLs. The system automatically creates a searchable knowledge base and allows users to ask questions grounded in the article content.

The application combines:

* Groq LLMs for answer generation
* LangChain for orchestration
* FAISS for vector search
* Sentence Transformers for embeddings
* Streamlit for the user interface

---

# 🚨 Problem Statement

Modern researchers, analysts, students, and professionals frequently rely on multiple online news sources to understand events and trends.

However, existing workflows have several limitations:

### 1. Copy-Pasting Articles into ChatGPT is Tedious

Users often need to:

* Open multiple articles
* Copy content manually
* Paste text into ChatGPT
* Repeat the process for every article

This becomes inefficient when dealing with multiple sources.

### 2. Lack of a Unified Knowledge Base

Information is distributed across different articles and websites.

Answering questions requires:

* Reading multiple sources
* Comparing information manually
* Keeping track of context

There is no centralized searchable repository.

### 3. Context Window Limitations

Large articles frequently exceed the context limits of LLMs.

As a result:

* Important information may be omitted
* Long articles need manual summarization
* Multi-document reasoning becomes difficult

### 4. Time-Consuming Research

Extracting insights from multiple articles can take significant effort.

Researchers need a system that can:

* Aggregate information
* Understand context
* Provide source-grounded answers

---

# 🎯 Solution

The News Research Tool automatically:

1. Accepts multiple news article URLs
2. Extracts article content
3. Splits text into manageable chunks
4. Generates vector embeddings
5. Stores embeddings in FAISS
6. Retrieves relevant context
7. Uses Groq LLMs to answer questions
8. Displays source references

This creates an intelligent knowledge base capable of answering questions across multiple articles.

---

# 🏗️ System Architecture

```text
User URLs
     │
     ▼
WebBaseLoader
     │
     ▼
Article Content
     │
     ▼
RecursiveCharacterTextSplitter
     │
     ▼
Text Chunks
     │
     ▼
Sentence Transformers
(all-MiniLM-L6-v2)
     │
     ▼
Vector Embeddings
     │
     ▼
FAISS Vector Store
     │
     ▼
Similarity Search
     │
     ▼
Relevant Chunks
     │
     ▼
Groq LLM
(llama-3.3-70b-versatile)
     │
     ▼
Answer + Sources
```

---

# ⚙️ Technology Stack

### Frontend

* Streamlit

### LLM

* Groq
* Llama 3.3 70B Versatile

### RAG Framework

* LangChain

### Vector Database

* FAISS

### Embedding Model

* Sentence Transformers
* all-MiniLM-L6-v2

### Data Processing

* WebBaseLoader
* RecursiveCharacterTextSplitter

### Environment Management

* Python Dotenv

---

# 🔄 Workflow

## Step 1: Input URLs

Users provide up to three article URLs.

## Step 2: Content Extraction

The application downloads and extracts article content.

## Step 3: Chunking

Articles are split into smaller chunks suitable for embedding.

## Step 4: Embedding Generation

Each chunk is converted into a numerical vector representation.

## Step 5: Vector Storage

Embeddings are stored in FAISS for efficient similarity search.

## Step 6: User Query

The user asks a question.

## Step 7: Retrieval

Relevant chunks are retrieved from FAISS.

## Step 8: Answer Generation

Groq generates a response using retrieved context.

## Step 9: Source Attribution

Sources are displayed alongside the answer.

---

# ✨ Key Features

✅ Multi-article research

✅ Retrieval-Augmented Generation (RAG)

✅ Source-grounded answers

✅ Vector similarity search

✅ Groq-powered inference

✅ Local embedding generation

✅ Streamlit user interface

✅ Fast retrieval using FAISS

---

# 📊 Example Use Cases

### Financial Research

* Compare coverage from multiple business publications
* Analyze company announcements

### Market Intelligence

* Aggregate industry news
* Extract competitive insights

### Academic Research

* Review multiple sources efficiently
* Identify key themes

### General News Analysis

* Ask questions across several news articles
* Generate concise summaries

---

# 📂 Project Structure

```text
news-research-tool/
│
├── main.py
├── .env
├── requirements.txt
├── faiss_store.pkl
├── README.md
│
└── assets/
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/snp-007/news-research-tool.git

cd news-research-tool
```

## Create Virtual Environment

```bash
py -3.11 -m venv venv

venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

## Run Application

```bash
streamlit run main.py
```

---

# 📈 Future Enhancements

* Support for unlimited URLs
* ChromaDB integration
* Citation highlighting
* PDF research support
* Article summarization mode
* Chat history memory
* Multi-language support
* Hybrid search (keyword + semantic)

---

# 🎓 Learning Outcomes

This project demonstrates practical experience with:

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Semantic Search
* Large Language Models
* Prompt Engineering
* Information Retrieval
* LangChain Workflows
* Streamlit Deployment

---

# 📜 License

This project is licensed under the MIT License.
