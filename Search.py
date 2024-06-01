import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.document_transformers import Html2TextTransformer
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import nest_asyncio
import os
import pandas as pd
from Scrape import scrape_jobs
from api import GEMINI_API_KEY, COHERE_API_KEY

os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

st.set_page_config(page_title="Job Search")

def main():
    st.header("Job Search Assistant 🔎")
    
    # Check session state for the button pressed
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = ""

    user_prompt = st.text_input("Enter your job search prompt", st.session_state['prompt'])

    if st.button("Search", key="search_button"):
        if user_prompt:
            st.session_state['prompt'] = user_prompt
            search_jobs(user_prompt)
        else:
            st.info("Please enter a job search prompt to see the available job listings.")
    
    st.header("💡Prompt Suggestions:")

    if st.button("Show me the companies hiring currently", key="hiring_button"):
        st.session_state['prompt'] = "companies hiring currently"
        search_jobs(st.session_state['prompt'])

    if st.button("Show me companies hiring currently, having rating 3.5+", key="rating_button"):
        st.session_state['prompt'] = "companies hiring currently, rating 3.5+"
        search_jobs(st.session_state['prompt'])

    if st.button("Show me the details of company: Accenture", key="details_button"):
        st.session_state['prompt'] = "details of company: Accenture"
        search_jobs(st.session_state['prompt'])

def search_jobs(user_prompt):
    job_listings = scrape_jobs()

    if job_listings.empty:
        st.warning("No job listings found.")
        return

    st.subheader("Your Job Search Prompt:")
    st.write(user_prompt)

    st.subheader("Job Listings:")
    st.write(job_listings)

    # formatting the data of the jobs listings.
    formatted_job_listings = [Document(
        page_content=f"Company Name: {row['name']}, Rating: {row['ratings']}, Category: {row['category']}",
        metadata={"name": row['name'], "ratings": row['ratings'], "category": row['category']}
    ) for _, row in job_listings.iterrows()]

    # Creating embeddings for the openings result.
    embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

    # Split the text in the stored data.
    text_splitter = RecursiveCharacterTextSplitter()
    split_docs = text_splitter.split_documents(formatted_job_listings)

    # Storing the embeddings into the vector store.
    vector = FAISS.from_documents(split_docs, embeddings)

    # Retrieving from the vector store.
    retriever = vector.as_retriever()

    llm = ChatGoogleGenerativeAI(model='gemini-pro')
    prompt = ChatPromptTemplate.from_template(
        """Answer the following questions based on only the provided contexts.
        <context>
        {context}
        </context>
        Question: {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    ret_response = retrieval_chain.invoke({"input": user_prompt})

    st.subheader("Response:")
    st.write(parse_response(user_prompt, ret_response["answer"]))

def parse_response(user_prompt, response):
    if "rating" in user_prompt.lower():
        return "\n".join([f"{i+1}. {line.split(': ')[1]}" for i, line in enumerate(response.split('\n')) if "Rating" in line])
    elif "details of company" in user_prompt.lower():
        return response
    else:
        return "\n".join([f"{i+1}. {line.split(': ')[1].split(',')[0]}" for i, line in enumerate(response.split('\n')) if "Company Name" in line])

if __name__ == "__main__":
    main()
