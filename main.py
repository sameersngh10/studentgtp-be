from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from pydantic import BaseModel
import os
from kb_search import user_input, generate_questions, get_pdf_text, get_text_chunks, get_vector_store, FAISS, GoogleGenerativeAIEmbeddings

class Item(BaseModel):
    message: str

app = FastAPI()

origins = ["http://localhost:4200"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store latest uploaded PDF path
latest_pdf_path = None

@app.get('/')
def read_root():
    return {"message": "App is running"}

@app.post("/getBotAnswer")
def read_item(item: Dict):
    global latest_pdf_path

    question = item['message']

    if latest_pdf_path is None:
        return {"error": "No PDF uploaded yet!"}

    response = user_input(question)  # This will now use the latest FAISS index
    return {"message": response}

@app.post("/generate-questions/")
async def generate_questions_api(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    pdf_text = get_pdf_text(file_path)
    questions = generate_questions(pdf_text)
    
    return {"questions": questions}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global latest_pdf_path

    # Save PDF file locally
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Update latest PDF path
    latest_pdf_path = file_path

    # Process PDF for embeddings
    pdf_text = get_pdf_text(file_path)
    text_chunks = get_text_chunks(pdf_text)
    get_vector_store(text_chunks)  # Generate embeddings automatically

    # Reload FAISS index after new embeddings are generated
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    FAISS.load_local('faiss_index', embeddings=embeddings, allow_dangerous_deserialization=True)

    return {"message": "PDF uploaded and embeddings updated successfully!", "pdf_path": file_path}
