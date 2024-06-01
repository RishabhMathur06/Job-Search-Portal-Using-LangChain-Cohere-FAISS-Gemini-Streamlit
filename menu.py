import streamlit as st
from Search import main as job_search_main
from ResumeAnalyzer import main as resume_analyzer_main
from RAG import main as jobmarketnews_main

def main():
    st.title("KnowledgeGEM 👨🏻‍💻")

    # Functionalities
    option = st.sidebar.selectbox(
        "Select an option",
        ("Job Search Assistant", "Resume Analyzer", "Job Market")
    )

    if option == "Job Search Assistant":
        job_search_main()  # Function to run the job search assistant
    elif option == "Resume Analyzer":
        resume_analyzer_main()  # Function to run the resume analyzer
    elif option == "Job Market":
        jobmarketnews_main()

    st.sidebar.markdown("---")  # Add a separator between navigation and main content

if __name__ == "__main__":
    main()