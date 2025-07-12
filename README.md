
# 🧠 Retrieval-Augmented Generation (RAG) Chatbot

This project implements a RAG (Retrieval-Augmented Generation) pipeline that combines traditional information retrieval with generative language modeling to build an intelligent chatbot capable of answering user queries with context-aware and document-grounded responses.

---

## 🔧 Project Architecture and Flow

```mermaid
graph TD
    A[Input Query] --> B[Preprocessing]
    B --> C[Embedding Generation]
    C --> D[Vector Store (FAISS/Chroma)]
    D --> E[Retriever]
    E --> F[Prompt Construction]
    F --> G[LLM (OpenAI/Local)]
    G --> H[Streaming Response]
```

- Preprocessing: Raw documents are cleaned and chunked for consistency.
- Embeddings: Chunks are converted into dense vector representations.
- Vector Store: Embeddings are stored using FAISS or Chroma for fast similarity search.
- Retriever: Top-k similar documents are retrieved for any user query.
- LLM: Retrieved docs + query are passed to a language model to generate a response.
- Streaming: Chatbot responds with streamed, token-by-token replies.

---

## 🚀 How to Run the Pipeline

### 1. Clone the Repository

```bash
git clone https://github.com/sameer23MCS120/Ai_Task_code.git
cd rag-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> If you use `langchain`, `openai`, `chromadb`, or `faiss`, ensure they are listed in `requirements.txt`.

---

## 📊 Preprocessing & Embedding Creation

### Step 1: Preprocess Documents

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader("data/my_docs.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
```

### Step 2: Create Embeddings

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embedding = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embedding)
db.save_local("faiss_index")
```

---

## 🧠 Model and Embedding Choices

| Component     | Choice                       | Notes |
|---------------|------------------------------|-------|
| Embeddings    | `OpenAIEmbeddings` / `HuggingFaceEmbeddings` | Depending on API or local preference |
| LLM           | `OpenAI GPT-4` / `Llama 3` / `Mistral`       | Choose between API or local inference |
| Vector Store  | FAISS / Chroma               | Efficient for large-scale retrieval |

---

## 💬 Running the Chatbot with Streaming

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

retriever = db.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(streaming=True, temperature=0.7)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

response = qa("What is the document about?")
for chunk in response["result"]:
    print(chunk, end="", flush=True)
```

> 💡 **Streaming requires** an LLM that supports streamed outputs (e.g., `ChatOpenAI(streaming=True)`).

---

## 🧪 Notebook

To reproduce all the above steps and run the chatbot interactively, open:

```bash
jupyter notebook Untitled19.ipynb
```

---

## 📂 Directory Structure

```
rag-chatbot/
│
├── data/                   # Input documents
├── faiss_index/            # Stored vector DB
├── Ai_Task.ipynb        # Main notebook
├── requirements.txt        # Project dependencies
└── README.md               # This file
``
