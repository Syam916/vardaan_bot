import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import time
import shutil

# Step 1: Load API Keys
# This function loads the API keys for OpenAI and Groq from the environment variables using the .env file.
def load_api_keys():
    load_dotenv()  # Load environment variables from the .env file
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")  # Set OpenAI API key
    groq_api_key = os.getenv('GROQ_API_KEY')  # Retrieve Groq API key
    return groq_api_key

# Step 2: Initialize the LLM Model
# This function initializes the Llama3 model using the Groq API key. The LLM will be used for document-based queries.
def initialize_llm(groq_api_key):
    return ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Step 3: Define the Prompt Template
# This function defines a prompt template, which is used to generate questions and answers based on the document context.
def create_prompt():
    return ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

# Step 4: Upload and Process PDF
# This function processes the uploaded PDF file by saving it into a directory and then using PyPDFLoader to extract text.
def process_pdf(uploaded_file):
    # Ensure the directory exists for storing uploaded documents
    if not os.path.exists("upload_documents"):
        os.makedirs("upload_documents")

    # Save the uploaded PDF into the "upload_documents" directory
    file_path = os.path.join("upload_documents", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and extract the content from the uploaded PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Step 5: Split PDF Text into Chunks
# This function splits the extracted text into smaller chunks (segments) for efficient vector embedding.
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Step 6: Generate Vector Embeddings
# This function creates vector embeddings for the text chunks using OpenAI embeddings, and stores them in a FAISS vector database.
def create_vector_store(final_documents):
    embeddings = OpenAIEmbeddings()  # Create embeddings for the document chunks
    vectors = FAISS.from_documents(final_documents, embeddings)  # Store in FAISS vector store for fast retrieval
    return vectors

# Step 7: Handle User Queries
# This function handles the user's input (query) and retrieves the most relevant document chunks from the vector store. It then generates a response using the LLM.
def handle_query(llm, prompt, vectors, user_query):
    # Create a chain that uses the LLM and the prompt template
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Use the vectors to create a retriever that searches through the document embeddings
    retriever = vectors.as_retriever()
    
    # Create the chain that combines retrieval and the LLM for answering the query
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure the time it takes to retrieve and generate the response
    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_query})
    end_time = time.process_time() - start

    return response, end_time

# Step 8: Clear Uploaded Documents
# This function deletes the "upload_documents" directory and its contents after the session ends to free up space.
def clear_upload_documents():
    if os.path.exists("upload_documents"):
        shutil.rmtree("upload_documents")  # Remove the entire directory

# Step 9: Main Application Flow
# This function sets up the main Streamlit interface, including PDF upload, text processing, and querying.
def main():
    # Step 1: Load API Keys for LLM
    groq_api_key = load_api_keys()

    # Step 2: Initialize LLM
    llm = initialize_llm(groq_api_key)

    # Step 3: Define the Prompt Template for question/answer generation
    prompt = create_prompt()

    # Step 4: Create the Streamlit interface with a title
    st.title("VIGNAN mini Pdf ChatBot")

    # Step 4.1: Add a subtitle below the title
    st.subheader("Powered by Vardaan Data Sciences")

    # Step 5: Allow users to upload PDF files
    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

    # If a PDF is uploaded, process it and generate vector embeddings
    if uploaded_pdf:
        # Step 6: Extract text from the uploaded PDF
        documents = process_pdf(uploaded_pdf)
        st.write("PDF uploaded and text extracted")

        # Step 7: Split the text into smaller chunks
        final_documents = split_documents(documents)
        st.write("Text split into smaller chunks")

        # Step 8: Create vector embeddings from the document chunks
        st.session_state.vectors = create_vector_store(final_documents)
        st.write("Vector Store DB is ready")


    # Step 9: Allow users to enter questions for document-based queries
    prompt1 = st.text_input("Enter Your Question From Documents")

    # Step 10: If the user has provided a question and the vector store exists, handle the query
    if prompt1 and 'vectors' in st.session_state:
        # Retrieve the relevant document chunks and generate a response
        response, response_time = handle_query(llm, prompt, st.session_state.vectors, prompt1)
        
        # Display the response and the time it took to generate it
        st.write(f"Response time: {response_time} seconds")
        st.write(response['answer'])

        # Debugging: Display the document chunks retrieved for the answer
        with st.expander("Document Similarity Search (Debug)"):
            for i, doc in enumerate(response["context"]):
                st.write(f"Retrieved Chunk {i+1}:")
                st.write(doc.page_content)
                st.write("--------------------------------")
    elif prompt1:
        st.warning("Please upload a PDF and process it first.")

# Call the main function to run the app
if __name__ == '__main__':
    main()

    # Clear uploaded documents when the session ends
    st.session_state.on_session_end = clear_upload_documents
