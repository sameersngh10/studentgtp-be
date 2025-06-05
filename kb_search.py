from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()
output_parser = StrOutputParser()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

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
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local('faiss_index')

def get_controversial_chain():
    prompt_template = """
    Your name is Friday, You are a helpful A+ certified assistant. Do not provide your reasoning for the answer.
    Just provide the answer to the user. Provide links to sites, forms, and contact information when providing information.
    If an answer is not available, just say "I'm still learning, therefore I don't currently have a response for the question you posed."
    
    Return all output in the form of HTML tags and make sure to highlight or bold key or important words and use new lines if required.
    
    Important: No other text formatting rather than HTML is accepted.
    
    Context: {context} 
    Question: \n {question} \n
    Answer:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = prompt | llm | output_parser
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local('faiss_index', embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_controversial_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question, "context": docs})
    
    return response

def generate_questions(pdf_text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(pdf_text)
    
    prompt_template = PromptTemplate.from_template(
        "Given the following document content:\n\n{content}\n\nGenerate 4 relevant questions a student might ask about this document."
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
    prompt = prompt_template.format(content=chunks[0])
    response = llm.invoke(prompt)
    
    return response.split("\n") if response else ["No questions generated."]


def main():
    pdf_path ="./pdf/hypervisor.pdf" 
    raw_text=get_pdf_text(pdf_path)
    text_chunks=get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    
    # while True:
    #     user_question=input("Ask from pdf:")
    #     if user_question:
    #         user_input(user_question)
    # user_question= st.text_input("Ask a question from the PDF Files & process it")
    # if user_question:
    #     user_input(user_question)
        
    # if st.button("Process PDF"):
    #     with st.spinner("Processing ...."):
    # raw_text=get_pdf_text(pdf_path)
    #     text_chunks=get_text_chunks(raw_text)
    #         get_vector_store(text_chunks)
            # st.success("Index Created Successfully")
    
if __name__=="__main__":
        main()