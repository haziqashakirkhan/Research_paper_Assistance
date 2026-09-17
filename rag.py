import os

import numpy as np

from dotenv import load_dotenv

from pypdf import PdfReader

import pdfplumber

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from the .env file."
    )


# ==================================================
# SETTINGS
# ==================================================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 6


# ==================================================
# GEMINI EMBEDDINGS & LLM
# ==================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GEMINI_API_KEY
)


# ==================================================
# IN-MEMORY DOCUMENT STORAGE
# ==================================================

document_chunks = []


# ==================================================
# TEXT CHUNKING
# ==================================================

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


# ==================================================
# PDF TEXT EXTRACTION
# ==================================================

def extract_pdf(file_path, file_name):
    """
    Extract text from a PDF page by page.

    PyPDF is used first.
    PDFplumber is used as a fallback.
    """

    pages = []

    # ------------------------------------------------
    # Open PDF using PyPDF
    # ------------------------------------------------

    reader = PdfReader(file_path)

    print(f"\nProcessing: {file_name}")
    print(f"Pages found: {len(reader.pages)}")

    # ------------------------------------------------
    # Open PDF using PDFplumber
    # ------------------------------------------------

    try:

        plumber_pdf = pdfplumber.open(file_path)

    except Exception as e:

        print(
            f"PDFplumber could not open the PDF: {e}"
        )

        plumber_pdf = None

    # ------------------------------------------------
    # Extract every page
    # ------------------------------------------------

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = ""

        # --------------------------------------------
        # Try PyPDF
        # --------------------------------------------

        try:

            text = page.extract_text() or ""

        except Exception as e:

            print(
                f"PyPDF failed on page "
                f"{page_number}: {e}"
            )

        # --------------------------------------------
        # PDFplumber fallback
        # --------------------------------------------

        if len(text.strip()) < 20 and plumber_pdf:

            try:

                plumber_page = (
                    plumber_pdf.pages[page_number - 1]
                )

                plumber_text = (
                    plumber_page.extract_text() or ""
                )

                if len(
                    plumber_text.strip()
                ) > len(text.strip()):

                    text = plumber_text

            except Exception as e:

                print(
                    f"PDFplumber failed on page "
                    f"{page_number}: {e}"
                )

        # --------------------------------------------
        # Store page text
        # --------------------------------------------

        text = text.strip()

        if text:

            pages.append(
                {
                    "text": text,
                    "page": page_number,
                    "source": file_name
                }
            )

    # ------------------------------------------------
    # Close PDFplumber
    # ------------------------------------------------

    if plumber_pdf:

        plumber_pdf.close()

    print(
        f"Pages with extractable text: "
        f"{len(pages)}"
    )

    return pages


# ==================================================
# ADD PDF
# ==================================================

def add_pdf(file_path, file_name):
    """
    Extract PDF text, split it into chunks,
    create embeddings, and store the chunks
    in memory.
    """

    global document_chunks

    pages = extract_pdf(
        file_path,
        file_name
    )

    # ------------------------------------------------
    # Check whether PDF contains text
    # ------------------------------------------------

    if not pages:

        raise ValueError(
            "No readable text was found in this PDF. "
            "It may be scanned or image-based."
        )

    # ------------------------------------------------
    # Remove existing chunks for this file to avoid duplicates
    # ------------------------------------------------

    document_chunks = [
        c for c in document_chunks if c["source"] != file_name
    ]

    total_chunks = 0

    # ------------------------------------------------
    # Process every page
    # ------------------------------------------------

    for page_data in pages:

        text = page_data["text"]

        page_number = page_data["page"]

        source = page_data["source"]

        # --------------------------------------------
        # Split page into chunks
        # --------------------------------------------

        chunks = split_text(text)

        # --------------------------------------------
        # Create embeddings
        # --------------------------------------------

        for chunk_index, chunk in enumerate(chunks):

            print(
                f"Creating embedding for "
                f"{source} | "
                f"Page {page_number} | "
                f"Chunk {chunk_index + 1}"
            )

            embedding = embeddings.embed_query(
                chunk
            )

            # ----------------------------------------
            # Store chunk
            # ----------------------------------------

            document_chunks.append(
                {
                    "text": chunk,
                    "source": source,
                    "page": page_number,
                    "chunk": chunk_index,
                    "embedding": embedding
                }
            )

            total_chunks += 1

    print(
        f"\nAdded {total_chunks} chunks "
        f"from {file_name}"
    )

    print(
        f"Total chunks in memory: "
        f"{len(document_chunks)}"
    )

    return total_chunks


# ==================================================
# COSINE SIMILARITY
# ==================================================

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    """

    a = np.array(
        vector_a,
        dtype=float
    )

    b = np.array(
        vector_b,
        dtype=float
    )

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:

        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ==================================================
# RETRIEVE RELEVANT DOCUMENTS
# ==================================================

def retrieve_documents(question, k=TOP_K):
    """
    Find the document chunks most relevant
    to the user's question.
    """

    # ------------------------------------------------
    # Empty question
    # ------------------------------------------------

    if not question.strip():

        return []

    # ------------------------------------------------
    # No documents
    # ------------------------------------------------

    if not document_chunks:

        print(
            "\nNo documents are currently loaded."
        )

        return []

    # ------------------------------------------------
    # Create question embedding
    # ------------------------------------------------

    query_embedding = embeddings.embed_query(question)

    # ------------------------------------------------
    # Calculate similarity scores
    # ------------------------------------------------

    scored_chunks = []

    for chunk_data in document_chunks:

        score = cosine_similarity(
            query_embedding,
            chunk_data["embedding"]
        )

        scored_chunks.append(
            (score, chunk_data)
        )

    # ------------------------------------------------
    # Sort by similarity score descending
    # ------------------------------------------------

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    # ------------------------------------------------
    # Select top K chunks
    # ------------------------------------------------

    top_chunks = [
        item[1] for item in scored_chunks[:k]
    ]

    return top_chunks


# ==================================================
# ASK QUESTION
# ==================================================

def ask_question(question):
    """
    Answer user's question based on retrieved research paper chunks using Gemini.
    """

    question = question.strip()

    if not question:

        return {
            "answer": "Please enter a valid question.",
            "sources": []
        }

    relevant_chunks = retrieve_documents(question, k=TOP_K)

    if not relevant_chunks:

        return {
            "answer": (
                "No documents are currently loaded or no relevant "
                "information was found. Please upload research paper PDFs first."
            ),
            "sources": []
        }

    # ------------------------------------------------
    # Build context string and source list
    # ------------------------------------------------

    context_blocks = []

    sources = []

    seen_sources = set()

    for idx, chunk in enumerate(relevant_chunks, start=1):

        context_blocks.append(
            f"--- Source [{idx}]: {chunk['source']} (Page {chunk['page']}) ---\n"
            f"{chunk['text']}"
        )

        source_key = (chunk["source"], chunk["page"])

        if source_key not in seen_sources:

            seen_sources.add(source_key)

            sources.append(
                {
                    "document": chunk["source"],
                    "page": chunk["page"]
                }
            )

    context_str = "\n\n".join(context_blocks)

    # ------------------------------------------------
    # Construct Prompt for Gemini
    # ------------------------------------------------

    prompt = (
        "You are an expert AI research assistant. Use the provided paper context below "
        "to answer the user's question accurately.\n"
        "Provide a clear, concise, and direct answer based strictly on the provided context. "
        "If the answer cannot be determined from the context, state that clearly.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = llm.invoke(prompt)

    answer_text = response.content

    if isinstance(answer_text, list):

        text_parts = [
            item["text"] for item in answer_text
            if isinstance(item, dict) and "text" in item
        ]

        answer_text = "\n".join(text_parts) if text_parts else str(answer_text)

    return {
        "answer": answer_text,
        "sources": sources
    }
