import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

#  Load API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", st.secrets.get("GOOGLE_API_KEY"))
if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY is missing! Add it in .env (local) or Streamlit secrets (Cloud).")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
    
genai.configure(api_key=GOOGLE_API_KEY)

#  Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

#  Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

#  Function to create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

#  Function to create Gemini AI conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as accurately as possible using the provided context. 
    If the answer is not available in the context, say: "The answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

#  Function to handle user input and query FAISS index
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        faiss_index_path = "faiss_index"

        #  Load FAISS index if it exists
        if not os.path.exists(faiss_index_path):
            st.error("⚠️ FAISS index not found! Please upload PDFs and process them first.")
            return

        new_db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

        #  Perform similarity search
        docs = new_db.similarity_search(user_question)

        #  Run the query through Gemini AI
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        #  Display response
        st.subheader("Reply:")
        st.write(response["output_text"])

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
        

#  Streamlit UI
def main():
    st.set_page_config(page_title="Chat with PDF - Gemini AI")
    st.header("💬 Chat with PDF using Gemini AI")

    #  User Input for Asking Questions
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload PDF Files")
        pdf_docs = st.file_uploader("Upload your PDFs and Click Submit", accept_multiple_files=True) 

        if st.button("📥 Submit & Process"):
            if not pdf_docs:
                st.error("⚠️ Please upload at least one PDF before submitting.")
            else:
                with st.spinner("🔄 Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Processing Complete! You can now ask questions.")
        st.title("💡 About the Chatbot")
        st.info("This is an AI-powered Applicaation that Chat with PDF using Gemini AI . No need to read PDFs just upload and get your answers!")
        st.markdown("🔹 **Developed by**: Mohit Manohar")  
        st.markdown("📢 **Instructions:**")
        st.markdown("- Upload the PDF in the SideBar")
        st.markdown("- Click on Process & Submit")
        st.markdown("- Enjoy an interactive chat experience! 😊")

#  Run the App
if __name__ == "__main__":
    main()
