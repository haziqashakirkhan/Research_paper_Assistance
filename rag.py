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
# GEMINI EMBEDDINGS
# ==================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)


# ==================================================
# IN-MEMORY DOCUMENT STORAGE
# ==================================================

# Every uploaded PDF is converted into chunks.
# Each chunk contains:
# - text
# - document name
# - page number
# - embedding

document_chunks = []


# ==================================================
# TEXT CHUNKING
# ==================================================

def split_text(text):
    """
    Split extracted PDF text into overlapping chunks.
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

    PyPDF is tried first.

    If PyPDF extracts very little text from a page,
    PDFplumber is used as a fallback.

    Each page keeps its page number and filename
    so the RAG system can later show sources.
    """

    pages = []

    # ----------------------------------------------
    # Open with PyPDF
    # ----------------------------------------------

    reader = PdfReader(file_path)

    print(f"\nProcessing: {file_name}")
    print(f"Pages found: {len(reader.pages)}")

    # ----------------------------------------------
    # Open PDFplumber once
    # ----------------------------------------------

    try:
        plumber_pdf = pdfplumber.open(file_path)

    except Exception as e:

        print(
            f"PDFplumber could not open the PDF: {e}"
        )

        plumber_pdf = None

    # ----------------------------------------------
    # Process every page
    # ----------------------------------------------

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = ""

        # ------------------------------------------
        # Try PyPDF
        # ------------------------------------------

        try:

            text = page.extract_text() or ""

        except Exception as e:

            print(
                f"PyPDF failed on page "
                f"{page_number}: {e}"
            )

        # ------------------------------------------
        # PDFplumber fallback
        # ------------------------------------------

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

        # ------------------------------------------
        # Save page if it contains text
        # ------------------------------------------

        text = text.strip()

        if text:

            pages.append(
                {
                    "text": text,
                    "page": page_number,
                    "source": file_name
                }
            )

    # ----------------------------------------------
    # Close PDFplumber
    # ----------------------------------------------

    if plumber_pdf:

        plumber_pdf.close()

    print(
        f"Pages with extractable text: "
        f"{len(pages)}"
    )

    return pages


# ===============================================
