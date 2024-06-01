#########################################################################################
#                                 LLM Setup                                             #
#########################################################################################

# Importing necessary libraries
import base64
import streamlit as st
from PIL import Image
import os
import io
import pdf2image
import google.generativeai as genai

# Calling the API from another file for convinence
from api import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

def main():
    def get_gemini_response(input, pdf_content, prompt):
        model = genai.GenerativeModel(model_name='gemini-pro-vision')
        # [input, pdf_content[0], prompt]
        response = model.generate_content([input, pdf_content[0], prompt])
        return response

    # Setup to alter the input file and convert it into suitable format
    # for further processing.
    def input_pdf_setup(uploaded_file):
        if uploaded_file is not None:
            # Convert the pdf into image
            images = pdf2image.convert_from_bytes(uploaded_file.read())

            first_page = images[0]

            # Convert to bytes
            img_byte_arr = io.BytesIO()
            first_page.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            pdf_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(img_byte_arr).decode()
                }
            ]
            
            return pdf_parts
        
        else:
            raise FileNotFoundError("No file uploaded.")
        
    #########################################################################################
    #                                 Streamlit App                                         #
    #########################################################################################

    st.header("ATS Resume Expert")
    input_text = st.text_area("Job Description: ", key="input")
    uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf"])

    if uploaded_file is not None:
        st.write("PDF Successfully Uploaded!")

    submit1 = st.button("Tell me about the Resume")
    submit2 = st.button("Compare Skills")
    submit3 = st.button("Percentage Match")

    input_prompt1 = """
        You are an experienced Technical HR Manager with technical experiecne in the field of any one of the following roles: 
        Data Science, Data Analyst, Machine Learning Engineer, AI Engineer, Full Stack developer, DevOPS, Web Development. 
        Your task is to review the provided resume against the job description for these above mentioned profiles. 
        Please share your professional evaluation on whether the candidate's profile aligns with the role. 
        Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
    """

    input_prompt2 = """
        You are an experienced Technical HR Manager with technical experiecne in the field of any one of the following roles: 
        Data Science, Data Analyst, Machine Learning Engineer, AI Engineer, Full Stack developer, DevOPS, Web Development. 
        Your task is to review the provided resume against the job description for these above mentioned profiles. 
        Analyze the resume very carefully and match the skills present in the user's resume with that present in Job
        Description. Show the output in two columns where the first column displays Matching Skills and second column displays
        Missing Skills.
    """

    input_prompt3 = """
        You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of Data Science, Data Analyst,
        Machine Learning Engineer, AI Engineer, Full Stack developer, DevOPS, Web Development and deep ATS functionality, 
        your task is to evaluate the resume against the provided job description. Give me the percentage of match if the 
        resume matches the job description. Output the matching percentage in this form: 
        "Resume Relevance Score: <print the score>" and then rate it into one of the three categories : Bad, Good or Excellent
        and display it.
    """

    if submit1:
        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_prompt1, pdf_content, input_text)
            for chunk in response:
                st.subheader("Response: ")
                st.write(chunk.text)

        else:
            st.write("Please upload the resume")

    if submit2:
        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_prompt2, pdf_content, input_text)
            for chunk in response:
                st.subheader("Response: ")
                st.write(chunk.text)

        else:
            st.write("Please upload the resume")

    if submit3:
        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_prompt3, pdf_content, input_text)
            for chunk in response:
                st.subheader("Response: ")
                st.write(chunk.text)

        else:
            st.write("Please upload the resume")

if __name__ == "__main__":
    main()