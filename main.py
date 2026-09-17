import os
import shutil

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from rag import add_pdf, ask_question


app = FastAPI(
    title="Research Paper Assistant",
    description="RAG-based research paper question answering system"
)


# -----------------------------
# Folders
# -----------------------------

UPLOAD_FOLDER = "/tmp/uploads" if os.getenv("VERCEL") else "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------
# Static files
# -----------------------------

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# -----------------------------
# Templates
# -----------------------------

templates = Jinja2Templates(
    directory="templates"
)


# -----------------------------
# Startup Event
# -----------------------------

@app.on_event("startup")
async def startup_event():
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                try:
                    add_pdf(file_path, filename)
                except Exception as e:
                    print(f"Error auto-loading {filename}: {e}")


# -----------------------------
# Home page
# -----------------------------

@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "answer": None,
            "sources": [],
            "message": None
        }
    )


# -----------------------------
# Upload PDFs
# -----------------------------

@app.post("/upload")
async def upload_papers(
    request: Request,
    files: list[UploadFile] = File(...)
):

    uploaded_files = []
    total_chunks = 0

    for file in files:

        # Check file format
        if not file.filename.lower().endswith(".pdf"):

            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "answer": None,
                    "sources": [],
                    "message": (
                        f"Unsupported file format: {file.filename}. "
                        "Please upload PDF files only."
                    )
                }
            )

        file_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        # Save PDF
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        try:

            chunks = add_pdf(
                file_path,
                file.filename
            )

            total_chunks += chunks

            uploaded_files.append(
                file.filename
            )

        except Exception as e:

            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "answer": None,
                    "sources": [],
                    "message": (
                        f"Could not process "
                        f"{file.filename}: {str(e)}"
                    )
                }
            )

    message = (
        f"Successfully uploaded "
        f"{len(uploaded_files)} paper(s) "
        f"and created {total_chunks} text chunks."
    )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "answer": None,
            "sources": [],
            "message": message
        }
    )


# -----------------------------
# Ask question
# -----------------------------

@app.post("/ask")
async def ask(
    request: Request,
    question: str = Form(...)
):

    question = question.strip()

    # Empty question
    if not question:

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "answer": None,
                "sources": [],
                "message": "Please enter a question."
            }
        )

    try:

        result = ask_question(question)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "answer": result["answer"],
                "sources": result["sources"],
                "message": None,
                "question": question
            }
        )

    except Exception as e:

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "answer": None,
                "sources": [],
                "message": (
                    f"Error while generating answer: {str(e)}"
                ),
                "question": question
            }
        )