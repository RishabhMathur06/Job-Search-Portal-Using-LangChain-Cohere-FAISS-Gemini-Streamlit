import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Libraries related to Web Scraping and transforming HTML text
from bs4 import BeautifulSoup
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader
# Libraries for Vector Embeddings Processes
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import nest_asyncio
import os

from api import GEMINI_API_KEY
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY

from api import COHERE_API_KEY

def main():
    st.header("Job Market News 📰")

    user_prompt = st.text_input("Want to know about the job market?")

    if st.button("Find News 🔎"):
        llm = ChatGoogleGenerativeAI(model='gemini-pro')
        
        # Creating a chat prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert news teller that shows latest news about Job Scenario in current market."),
            ("user", user_prompt)
        ])
        
        nest_asyncio.apply()

        # For providing more context to the LLM for generating resposne,
        #  we would provide it data using external source such as websites, docs, etc.
        websites = ["https://economictimes.indiatimes.com/topic/indian-job-market",
                "https://www.bloombergquint.com/topic/indian-job-market",
                "https://www.livemint.com/topic/indian-job-market",
                "https://www.cnbc.com/jobs/",
                "https://content.techgig.com/trending-news"]

        # Scrapes the above websites.
        loader = AsyncChromiumLoader(websites)
        docs = loader.load()

        # Convert the HTML to plain text
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)


        # Creating a vector embedding model that makes vector embeddings
        #  and store it in vector store.
        # First creating a "embedding model."
        embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
        # Building "FAISS" vector index, where documents would be ingested
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs_transformed)
        vector = FAISS.from_documents(documents, embeddings)

        # Now, creating a retrieval chain using which,
        #  we would extract information with respect to the vector 
        #  embeddings stored in vector store and LLM response.
        prompt = ChatPromptTemplate.from_template(
            """ Answer the following questions based on only the provided contexts.
                <context>
                {context}
                </context>
                
                Question: {input}
            """
        )

        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Creating a ":String Output Parser" to convert:
        #      ChatMessage -> String Output
        output_parser = StrOutputParser()

        # Creating a LLM Chain to enhance output
        chain = prompt | llm | output_parser

        # Fetching the response, (normal chain)
        ## response = chain.invoke({"input": user_prompt})

        # Retriever response
        ret_response = retrieval_chain.invoke({"input": user_prompt})


        st.write("Response:", ret_response["answer"])

if __name__ =="__main__":
    main()