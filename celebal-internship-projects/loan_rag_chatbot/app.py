# AIzaSyAM5ROazVzYQTyWroC76oROYOfoXypTgog
# venv\Scripts\activate
# streamlit run app.py

import os

# ✅ Must be set before importing any gRPC clients
os.environ["GRPC_POLL_STRATEGY"] = "completion_queue"

import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# --- Configuration ---
# IMPORTANT: Replace "YOUR_GEMINI_API_KEY" with your actual Gemini API key.
# For local development, it's recommended to use environment variables or a .env file.
# Example using os.environ:
# os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"
# For this Canvas environment, we'll directly assign it for demonstration.
# In a real application, never hardcode API keys.
GOOGLE_API_KEY = "AIzaSyAM5ROazVzYQTyWroC76oROYOfoXypTgog" # Your Gemini API key has been inserted here.

# --- Fix for RuntimeError: No current event loop ---
# This environment variable forces gRPC to use a synchronous polling strategy,
# which helps avoid the "no current event loop" error in Streamlit's context.
# 'completion_queue' is suitable for Windows.
os.environ["GRPC_POLL_STRATEGY"] = "completion_queue"
# --- End Fix ---

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Loan Approval Q&A Chatbot", layout="centered")
st.title("🏡 Loan Approval Q&A Chatbot")
st.markdown("""
Ask me anything about loan approvals based on the provided dataset!
""")

# --- Data Loading and Preprocessing ---
@st.cache_data # Cache the data loading to improve performance
def load_and_process_data(file_path):
    """Loads the CSV, preprocesses it, and converts rows to text documents."""
    try:
        df = pd.read_csv(file_path)

        # Handle missing values: Fill numerical with median, categorical with mode
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

        # Convert each row into a descriptive text document
        documents = []
        for index, row in df.iterrows():
            doc_string = f"Loan Application Details (ID: {row['Loan_ID']}):\n" \
                         f"Gender: {row['Gender']}, Married: {row['Married']}, Dependents: {row['Dependents']},\n" \
                         f"Education: {row['Education']}, Self_Employed: {row['Self_Employed']},\n" \
                         f"Applicant Income: {row['ApplicantIncome']}, Coapplicant Income: {row['CoapplicantIncome']},\n" \
                         f"Loan Amount: {row['LoanAmount']}, Loan Amount Term: {row['Loan_Amount_Term']},\n" \
                         f"Credit History: {row['Credit_History']}, Property Area: {row['Property_Area']},\n" \
                         f"Loan Status: {row['Loan_Status']}"
            documents.append(doc_string)
        return documents
    except FileNotFoundError:
        st.error(f"Error: Dataset file not found at '{file_path}'. Please ensure 'Training Dataset.csv' is in the same directory as 'app.py'.")
        return []
    except Exception as e:
        st.error(f"An error occurred during data loading or processing: {e}")
        return []

# --- RAG Model Initialization ---
@st.cache_resource # Cache the model initialization
def setup_rag_model(documents):
    """Sets up the embedding model, vector store, and RAG chain."""
    if not documents:
        return None, None

    # Initialize Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    # --- Persistent Chroma Vector Store ---
    # Define the directory where the vector store will be saved
    persist_directory = "./chroma_db"
    collection_name = "loan_applications"

    # Check if the persistent directory already exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        st.info("Loading existing vector store from disk...")
        # Load the existing vector store
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        st.success("Vector store loaded successfully!")
    else:
        st.info("Creating new vector store from loan data... This may take a moment.")
        # Create a new Chroma vector store from the documents
        vectorstore = Chroma.from_texts(
            documents,
            embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        # Persist the new vector store to disk
        vectorstore.persist()
        st.success("New vector store created and saved to disk!")
    # --- End Persistent Chroma Vector Store ---

    # Initialize the ChatGoogleGenerativeAI model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

    # Define the prompt template for the RAG chain
    # This prompt guides the LLM on how to use the retrieved context.
    prompt_template = """
    You are a helpful AI assistant specialized in loan application data.
    Use the following context to answer the user's question.
    If you don't know the answer based on the provided context, politely state that you don't have enough information.
    Do not make up answers.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create a RetrievalQA chain
    # 'stuff' chain type puts all retrieved documents into the prompt.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 most relevant documents
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain, vectorstore

# --- Main Application Logic ---
def main():
    # Load and process data
    documents = load_and_process_data('Training Dataset.csv')

    # Setup RAG model
    qa_chain, vectorstore = setup_rag_model(documents)

    if qa_chain is None:
        st.warning("Chatbot cannot be initialized without data. Please check the dataset file.")
        return

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about loan approvals..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from RAG chain
                    response = qa_chain({"query": prompt})
                    answer = response["result"]
                    source_documents = response.get("source_documents", [])

                    st.markdown(answer)

                    # Display source documents (optional, for debugging/transparency)
                    if source_documents:
                        with st.expander("See Retrieved Sources"):
                            for i, doc in enumerate(source_documents):
                                st.write(f"**Source {i+1}:**")
                                st.text(doc.page_content)
                                st.markdown("---")

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"An error occurred while generating response: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})

if __name__ == "__main__":
    main()