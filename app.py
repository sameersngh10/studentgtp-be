from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "student-gpt-index"

# Create the index if it doesn't already exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Make sure this matches your embedding model output size
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Fetch the index
index = pc.fetch_index(index_name)

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(docs)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Convert chunks to embeddings and upload to Pinecone
    vectors = [embeddings.embed(chunk) for chunk in text_chunks]
    ids = [str(i) for i in range(len(vectors))]
    index.upsert(vectors=list(zip(ids, vectors)))
    
def get_controversial_chain():
    prompt_template = """
    Your name is Friday, a helpful A+ certified assistant. Do not provide reasoning, just answers.
    Provide links, forms, and contact information where helpful.
    If unsure, respond with "I'm still learning, therefore I don't currently have a response for the question you posed."
    Answer in HTML format.

    Context: {context}
    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Convert question to embedding and search in Pinecone
    question_vector = embeddings.embed(user_question)
    search_results = index.query(question_vector, top_k=5)
    docs = [result["metadata"]["text"] for result in search_results["matches"]]

    # Process with Gemini-pro chain
    chain = get_controversial_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Display the response
    st.write("Friday:", response["output_text"])

def main():
    st.set_page_config(page_title="Chat With PDF Using Gemini")
    st.header("GGITS BOT")
    st.header("Gyan Ganga AI Companion")

    pdf_path = "./pdf/GGITSpdf.pdf"
    if st.button("Process PDF"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_path)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Index Created Successfully")

    user_question = st.text_input("Ask a question from the PDF files & process it")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
