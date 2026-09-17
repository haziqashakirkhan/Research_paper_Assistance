import os
import hashlib

from dotenv import load_dotenv
from pypdf import PdfReader
import pdfplumber

import chromadb

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from the .env file.")


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "research_papers"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 6


# --------------------------------------------------
# GEMINI EMBEDDINGS
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)


# --------------------------------------------------
# CHROMA DATABASE
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def split_text(text):
    """
    Split text into overlapping chunks.
    """

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------

def extract_pdf(file_path, file_name):
    """
    Extract text page-by-page.

    First tries PyPDF.
    If a page has little/no text, PDFplumber is used
    as a fallback.
    """

    pages = []

    reader = PdfReader(file_path)

    print(f"\nProcessing: {file_name}")
    print(f"Pages found: {len(reader.pages)}")

    for page_number, page in enumerate(reader.pages, start=1):

        text = ""

        # ------------------------------------------
        # Try PyPDF
        # ------------------------------------------

        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(
                f"PyPDF failed on page {page_number}: {e}"
            )

        # ------------------------------------------
        # PDFplumber fallback
        # ------------------------------------------

        if len(text.strip()) < 20:

            try:

                with pdfplumber.open(file_path) as pdf:

                    plumber_page = pdf.pages[page_number - 1]

                    plumber_text = (
                        plumber_page.extract_text() or ""
                    )

                    if len(plumber_text.strip()) > len(
                        text.strip()
                    ):
                        text = plumber_text

            except Exception as e:

                print(
                    f"PDFplumber failed on page "
                    f"{page_number}: {e}"
                )

        text = text.strip()

        if text:

            pages.append(
                {
                    "text": text,
                    "page": page_number,
                    "source": file_name
                }
            )

    print(f"Pages with extractable text: {len(pages)}")

    return pages


# --------------------------------------------------
# ADD PDF TO VECTOR DATABASE
# --------------------------------------------------

def add_pdf(file_path, file_name):

    pages = extract_pdf(
        file_path,
        file_name
    )

    if not pages:

        raise ValueError(
            "No readable text was found in this PDF. "
            "It may be scanned or image-based."
        )

    total_chunks = 0

    for page_data in pages:

        text = page_data["text"]

        page_number = page_data["page"]

        source = page_data["source"]

        chunks = split_text(text)

        for chunk_index, chunk in enumerate(chunks):

            # Create unique ID
            raw_id = (
                f"{source}_"
                f"{page_number}_"
                f"{chunk_index}_"
                f"{chunk}"
            )

            chunk_id = hashlib.md5(
                raw_id.encode("utf-8")
            ).hexdigest()

            # Create embedding
            embedding = embeddings.embed_query(
                chunk
            )

            collection.add(
                ids=[chunk_id],

                embeddings=[embedding],

                documents=[chunk],

                metadatas=[
                    {
                        "source": source,
                        "page": page_number,
                        "chunk": chunk_index
                    }
                ]
            )

            total_chunks += 1

    print(
        f"Added {total_chunks} chunks "
        f"from {file_name}"
    )

    return total_chunks


# --------------------------------------------------
# RETRIEVE RELEVANT DOCUMENTS
# --------------------------------------------------

def retrieve_documents(question, k=TOP_K):

    if not question.strip():

        return []

    count = collection.count()

    print(
        f"\nVector database contains "
        f"{count} chunks."
    )

    if count == 0:

        return []

    # Don't request more documents than exist
    k = min(k, count)

    query_embedding = embeddings.embed_query(
        question
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    retrieved = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        retrieved.append(
            {
                "text": document,
                "source": metadata.get(
                    "source",
                    "Unknown document"
                ),
                "page": metadata.get(
                    "page",
                    "Unknown"
                )
            }
        )

    print(
        f"Retrieved {len(retrieved)} chunks "
        f"for question: {question}"
    )

    return retrieved


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

def ask_question(question):

    documents = retrieve_documents(
        question
    )

    # ----------------------------------------------
    # No documents uploaded
    # ----------------------------------------------

    if not documents:

        return {
            "answer": (
                "The answer is not available "
                "in the uploaded documents."
            ),
            "sources": []
        }

    # ----------------------------------------------
    # Build context
    # ----------------------------------------------

    context_parts = []

    sources = []

    for document in documents:

        source = document["source"]

        page = document["page"]

        text = document["text"]

        context_parts.append(
            f"""
Document: {source}
Page: {page}

Content:
{text}
"""
        )

        source_info = {
            "document": source,
            "page": page
        }

        if source_info not in sources:

            sources.append(
                source_info
            )

    context = "\n\n----------------\n\n".join(
        context_parts
    )

    # ----------------------------------------------
    # RAG Prompt
    # ----------------------------------------------

    prompt = f"""
You are a Research Paper Assistant.

Your job is to answer questions using ONLY
the information contained in the uploaded
research papers provided in the CONTEXT.

IMPORTANT RULES:

1. Do NOT use outside knowledge.

2. Do NOT make up information.

3. Do NOT guess.

4. You may summarize information from the
   provided context.

5. You may combine information from multiple
   retrieved sections if the context supports it.

6. If the requested information cannot be
   found in the context, respond exactly:

"The answer is not available in the uploaded documents."

7. When answering, be clear and concise.

8. For questions asking for a summary, explain
   the main topic, problem, methodology,
   findings, and other relevant information
   ONLY when those details are present.

CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # ----------------------------------------------
    # Gemini
    # ----------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

    response = llm.invoke(prompt)

    # ----------------------------------------------
    # Handle Gemini response
    # ----------------------------------------------

    content = response.content

    if isinstance(content, list):

        answer = "\n".join(
            (
                item.get("text", "")
                if isinstance(item, dict)
                else str(item)
            )
            for item in content
        )

    else:

        answer = str(content)

    answer = answer.strip()

    # ----------------------------------------------
    # If answer unavailable
    # ----------------------------------------------

    if (
        "not available in the uploaded documents"
        in answer.lower()
    ):

        sources = []

    return {
        "answer": answer,
        "sources": sources
    }