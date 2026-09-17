````markdown
# Research Paper Assistant

An AI-powered Research Paper Assistant that uses **Retrieval-Augmented Generation (RAG)** to answer questions based only on information found in uploaded research papers.

Users can upload one or multiple PDF research papers and ask questions about their content. The application retrieves the most relevant sections from the uploaded documents before generating an answer with Gemini.

## Features

- Upload one or multiple research paper PDFs
- Extract text using PyPDF and PDFplumber
- Split documents into smaller text chunks
- Generate Gemini embeddings for document chunks
- Retrieve relevant chunks using cosine similarity
- Generate answers using Google Gemini
- Answer questions using only uploaded document content
- Display source document name and page number
- Handles empty questions and unsupported file formats
- Provides a clear response when information is not found

## How RAG Works

The application follows this pipeline:

```text
Research Paper PDF
        ↓
Text Extraction
(PyPDF / PDFplumber)
        ↓
Text Chunking
        ↓
Gemini Embeddings
        ↓
In-Memory Vector Storage
        ↓
Cosine Similarity Search
        ↓
Relevant Document Chunks
        ↓
Gemini
        ↓
Answer + Sources
````

Instead of sending the entire research paper to the LLM, the application first retrieves the sections that are most relevant to the user's question.

## Tech Stack

* **Python**
* **FastAPI**
* **PyPDF**
* **PDFplumber**
* **NumPy**
* **LangChain**
* **Google Gemini**
* **HTML**
* **CSS**
* **JavaScript**

## Project Structure

```text
research_paper_assistance/
│
├── static/
│   ├── scripts.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── uploads/
│
├── .env
├── .gitignore
├── main.py
├── rag.py
└── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/haziqashakirkhan/Research_paper_Assistance.git
```

Move into the project:

```bash
cd Research_paper_Assistance
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add your Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Do not upload your `.env` file to GitHub.

### 5. Run the application

```bash
python -m uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Example Questions

After uploading a research paper, you can ask questions such as:

* What problem does this paper address?
* Summarize the paper.
* What methodology was used?
* Which dataset was used?
* What algorithms were used?
* What are the key findings?
* What are the limitations?
* What future work is suggested?

## RAG Approach

The application does not directly ask Gemini to answer from general knowledge.

Instead:

1. The PDF is extracted page by page.
2. The extracted text is divided into chunks.
3. Each chunk is converted into an embedding.
4. The user's question is also converted into an embedding.
5. Cosine similarity is used to compare the question with document chunks.
6. The most relevant chunks are retrieved.
7. Only the retrieved context is provided to Gemini.
8. Gemini generates the final answer from that context.
9. The relevant document names and page numbers are returned as sources.

## Handling Missing Information

If the requested information cannot be found in the uploaded documents, the assistant responds:

> "The answer is not available in the uploaded documents."

This helps prevent the assistant from presenting unrelated information as if it came from the research papers.

## Important Note

The current implementation stores document chunks and embeddings **in memory**.

This makes the project simple and suitable for learning and demonstration, but uploaded documents are not permanently stored in a production database.

## Future Improvements

* Persistent vector storage
* Support for scanned PDFs using OCR
* Better chunking based on sections and paragraphs
* Streaming responses
* Conversation history
* Improved source highlighting
* User authentication
* Production-ready document storage

## Learning Goals

This project was built to understand the practical workflow of a RAG application, including:

* PDF document processing
* Text chunking
* Embeddings
* Semantic retrieval
* Vector similarity
* Prompt grounding
* LLM integration
* FastAPI backend development

## Author

**Haziqa Shakir Khan**

AI & Data Science Student
Aspiring AI Engineer

GitHub: [haziqashakirkhan](https://github.com/haziqashakirkhan)

```

This version is intentionally **clean and recruiter-friendly** rather than packed with badges, emojis, or unnecessary sections.
```
