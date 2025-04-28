import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import io
import re
import json
import docx
import PyPDF2
import streamlit as st
import pandas as pd
from io import BytesIO
from streamlit_chat import message
from langchain_core.prompts import PromptTemplate
load_dotenv()
llm = ChatGroq(temperature=1,
             model_name=st.secrets["MODEL_NAME"],
             api_key=st.secrets["GROQ_API_KEY"],
             model_kwargs={"response_format": {"type": "json_object"}})

if 'json_data' not in st.session_state:
    st.session_state.json_data = []
if 'markdown_data' not in st.session_state:
    st.session_state.markdown_data = []
if 'match_results' not in st.session_state:
    st.session_state.match_results = []

# Function to extract text from PDF
#mixtral-8x7b-32768
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text().strip()
    return text

# Function to extract text from DOCX  
def extract_text_from_doc(file):
    doc = docx.Document(io.BytesIO(file.read()))
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text.strip() + "\n"
    return text

# Function to extract required information from resume text
def extract_text_from_file(file):
    file_type = file.name.split(".")[-1]
    if file_type == "pdf":
        return extract_text_from_pdf(file)
    elif file_type == "docx":
        return extract_text_from_doc(file)

# function to extract JSON from the resume text    
def extract_json_from_text(resume_text):

    format_instruction = """
        {
        "name": "Candidate's full name",
        "email": "Candidate's email address",
        "phone_number": "Candidate's phone number",
        "location": {
            "address": "Candidate's full address",
            "city": "Candidate's city",
            "country": "Candidate's country"
        },
        "linkedin_profile": "URL to LinkedIn profile",
        "github_profile": "URL to GitHub profile (if applicable)",
        "portfolio_website": "URL to personal portfolio or website (if applicable)",
        "career_objective": "Candidate's career objective or summary",
        "total_experience": "Total years of work experience",
        "relevant_experience": "Years of relevant experience",
        "current_job_title": "Candidate's current job title",
        "current_company": "Candidate's current company",
        "previous_job_titles": [
            "List of previous job titles"
        ],
        "previous_companies": [
            "List of previous companies"
        ],
        "skills": {
            "technical_skills": [
            "List of technical skills"
            ],
            "soft_skills": [
            "List of soft skills"
            ]
        },
        "education": [
            {
            "degree": "Degree obtained",
            "institution": "Institution name",
            "year_of_passing": "Year of passing",
            "division": "Division/Grade/CGPA"
            }
        ],
        "certifications": "List of certifications (if any)",
        "projects": [
            {
            "project_name": "Name of the project",
            "description": "Brief description of the project",
            "technologies_used": "List of technologies/tools used",
            "role": "Role in the project"
            }
        ],
        "achievements": "List of major achievements (if any)",
        "publications": "List of publications (if applicable)",
        "languages": [
            "List of languages"
        ],
        }
    """

    prompt_template = """
        You are tasked with extracting data from resume and returning a JSON structre.
        {format_instruction}
        Resume Text:
        {resume_text}
    """

    prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["format_instruction", "resume_text"],
    )
    
    llm_resume_parser = prompt | llm
    
    parsed_candidate_data = llm_resume_parser.invoke({"format_instruction": format_instruction, "resume_text":resume_text})
    
    return json.loads(parsed_candidate_data.content)

# function to match JD with the resume and return match score alongwith justifications
def match_JD_with_resume(JD_text, candidate_json):
    
    job_description = JD_text
    candidate = candidate_json
    
    prompt_template = """
        You are tasked with matching the Job Description provided with the candidate's JSON data and returning a match score out of 100. After scoring, you must determine the Application Status based on this logic: if the score is less than or equal to 50, the status should be "Rejected"; if greater than 60, it should be "Shortlisted".

        Please return the results strictly in the following format without any additional explanations:
        Match Score: <score>
        Application Status: <status>

        Score Breakdown:
        1. <Reason_Title> - <One-line explanation>
        2. <Reason_Title> - <One-line explanation>
        3. <Reason_Title> - <One-line explanation>

        Job Description:
        {job_description}

        Candidate's JSON:
        {candidate}
    """
    
    prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["job_description"],
    )
    
    llm_jd_match = prompt | llm
    
    resume_JD_match = llm_jd_match.invoke({"candidate":candidate, "job_description":job_description})
    
    return resume_JD_match.content

# Function to load JD
def load_jd(file):
    return extract_text_from_file(file=file)

# Function to load JSON files
def load_json_files(directory='Output/JSON'):
    json_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_list.append(data)
    return json_list

# Function to save JSON data
def save_json(data, filename, directory='Output/JSON'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    st.session_state.json_data.append(filepath)
    
# Function to save Match Score data for individual candidates
def save_evaluation_to_markdown(data, filename, directory='Output/Evaluations'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(data)
    # Store the file path in session state for tracking
    if 'markdown_data' not in st.session_state:
        st.session_state.markdown_data = []
    st.session_state.markdown_data.append(filepath)
    
def extract_name(user_input):
    # Simple extraction logic, assuming the name is the first part of the input
    # This can be improved based on specific requirements
    return user_input.split()[0] if user_input else None

# Function to extract Match Score and Application Status from markdown files
def extract_evaluation_from_markdown(filepath):

    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    # Use regex to extract Match Score and Application Status
    match_score = re.search(r"Match Score:\s*(\d+)", content)
    application_status = re.search(r"Application Status:\s*(\w+)", content)

    # Get values if found, else default to 'N/A'
    score = match_score.group(1) if match_score else None
    status = application_status.group(1) if application_status else None

    return {
        "File Name": os.path.basename(filepath),
        "Match Score": score,
        "Application Status": status
    }

 #Function to process all markdown files and extract required information
def process_markdown_files(directory='Output/Evaluations'):
    data = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.md'):
                filepath = os.path.join(directory, filename)
                extracted_data = extract_evaluation_from_markdown(filepath)
                data.append(extracted_data)
    return data

# Streamlit UI
# setting up the page header here.
hide_st_style = """
                <style>
                #MainMenu {visibility : hidden;}
                header {visibility : hidden;}
                </style>
                """

st.set_page_config(
    page_title="GenAI Resume-JD Parser",
    page_icon="📃",
    layout="wide"  # Set layout to wide for better UI experience
)


# removing all the default streamlit configs here
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #E8F1F5 0%, #B3E0F2 25%, #84CEE4 50%, #59A5D8 75%, #386FA4 100%);
    }

    /* For better readability on lighter background */
    .stMarkdown, .stText, h1, h2, h3 {
        color: #333333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Option 4: 3D Box Effect
st.markdown("""<div style="background: white; padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; transform: perspective(1000px) rotateX(5deg); box-shadow: 0 5px 15px rgba(0,0,0,0.2), 0 15px 35px rgba(0,0,0,0.1); transition: transform 0.3s ease;"><h1 style="background: linear-gradient(45deg, #2c3e50, #3498db); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Poppins', sans-serif; font-size: 2.5rem; font-weight: 700; margin: 0; padding: 10px;">Resume Parser (Hire Mate)</h1></div>""", unsafe_allow_html=True)

# Sidebar for Job Description Upload
st.sidebar.header("Job Description 📃")
jd_file = st.sidebar.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"], key = "jd_file")

if jd_file is not None:
    jd_text = load_jd(jd_file)
    st.sidebar.success("Job Description Processed Successfully!")
else:
    jd_text = ""
    st.sidebar.warning("Upload a Job Description to get started")
    
st.header("Upload Candidate Resumes 📂")
uploaded_files = st.file_uploader("choose your PDF or DOCX file",type=["pdf", "docx"], accept_multiple_files=True)
st.markdown("""<style>.hover-column:hover {transform: translateY(-5px); background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(5px); border: 2px solid rgba(255, 255, 255, 0.2); box-shadow: 0 8px 25px rgba(0, 123, 255, 0.2);} .stButton > button {width: 100%; padding: 15px 20px; border-radius: 12px; font-weight: 600; font-size: 16px; color: white; background: linear-gradient(45deg, #007bff, #00bfff); border: none; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0, 123, 255, 0.2);} .stButton > button:hover {transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0, 123, 255, 0.3); background: linear-gradient(45deg, #0056b3, #0098ff);} .stProgress > div > div {background: linear-gradient(to right, #007bff, #00bfff); border-radius: 10px;} .stSuccess {border-radius: 10px; animation: successPulse 2s infinite;} @keyframes successPulse {0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); } 100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }}</style>""", unsafe_allow_html=True)

col_1, col_2, col_3, col_4 = st.columns(4)

with col_1:
    st.markdown('<div class="hover-column">', unsafe_allow_html=True)
    if st.button("Process Resumes with AI 🧠"):
        if not uploaded_files:
            st.error("Please upload at least one resume.")
        elif not jd_file:
            st.error("Please upload a Job Description in the sidebar.")
        else:
            with st.spinner('Processing resumes...'):
                progress_bar = st.progress(0)
                total_files = len(uploaded_files)

                for idx, file in enumerate(uploaded_files):
                    text = extract_text_from_file(file)
                    extracted_info = extract_json_from_text(text)

                    # ✅ Use FULL WIDTH container for JSON
                    with st.container():
                        with st.expander("Processed JSON from Resume Files 🧠🪄", expanded=True):
                            # ✅ Custom CSS to stretch expander width
                            st.markdown(
                                """
                                <style>
                                .st-emotion-cache-16txtl3 {
                                    width: 100% !important;
                                }
                                </style>
                                """,
                                unsafe_allow_html=True
                            )
                            st.json(extracted_info)

                    filename = os.path.splitext(file.name)[0] + "_data.json"
                    save_json(extracted_info, filename)

                    progress_bar.progress((idx + 1) / total_files)

            st.success(f"{len(uploaded_files)} resumes processed & saved successfully.")
            st.balloons()
    st.markdown('</div>', unsafe_allow_html=True)
with col_4:
    st.markdown('<div class="hover-column">', unsafe_allow_html=True)
    if st.button("Flush Loaded Data 🗑"):
        st.session_state.json_data = []
        st.session_state.markdown_data = []
        st.session_state.match_results = []
        if 'jd_file' in st.session_state:
            del st.session_state.jd_file
        st.success("All data has been cleared 👍")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col_3:
    st.markdown('<div class="hover-column">', unsafe_allow_html=True)
    if st.button("Download Evaluated Data"):
        if not st.session_state.markdown_data:
            st.warning("Please Apply for Evaluation first ⚠")
        else:
            # Process the markdown files and extract data
            evaluation_data = process_markdown_files()
            if evaluation_data:
                # Convert data to DataFrame
                df = pd.DataFrame(evaluation_data)
                with st.expander("Preview Dataframe"):
                    st.dataframe(df)
                # Create an Excel file in memory
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Evaluations')
                excel_buffer.seek(0)

                # Download button for the Excel file
                st.download_button(
                    label="Download Excel",
                    data=excel_buffer,
                    file_name="evaluation_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No Evaluated Files Found ⚠")
    st.markdown('</div>', unsafe_allow_html=True)

with col_2:
    st.markdown('<div class="hover-column">', unsafe_allow_html=True)
    # Matching JD with Resumes and Generating Excel
    if st.button("Evaluate Resumes w.r.t. Job 💯"):
        if not jd_text:
            st.error("Please upload a Job Description in the sidebar.")
        elif not st.session_state.json_data:
            st.error("No processed JSON files found. Please process resumes first.")
        else:
            if not st.session_state.match_results:  # Only run if no match results are present
                with st.spinner('Matching resumes with Job Description...'):
                    total_files = len(st.session_state.json_data)
                    progress_bar = st.progress(0)
                    for idx, json_file in enumerate(st.session_state.json_data):
                        with open(json_file, 'r', encoding='utf-8') as f:
                            candidate_data = json.load(f)
                            match = match_JD_with_resume(jd_text, candidate_data)
                            st.session_state.match_results.append(match)
                        # Update progress bar
                        progress = (idx + 1) / total_files
                        progress_bar.progress(progress)
                    
                with st.spinner('Saving the Evaluated Resumes...'):
                    for idx, file in enumerate(uploaded_files):
                        filename_evaluated = os.path.splitext(file.name)[0] + "_evaluated.md"
                        save_evaluation_to_markdown(data=st.session_state.match_results[idx], filename=filename_evaluated)
                    st.success("All resumes have been evaluated successfully and saved.")
                    st.balloons()
    st.markdown('</div>', unsafe_allow_html=True)

# Display processed JSON files
if st.session_state.json_data:
    with st.expander("Processed JSON from Resume Files 🧠🪄", expanded=False):
        for json_file in st.session_state.json_data:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                st.json(data)

# Display the Processed Match Results.
if st.session_state.match_results:
    with st.expander("Preview Resume Match Results with Job Description 🔍🪄", expanded=False):
        st.write(st.session_state.match_results)
from langchain.schema import HumanMessage, AIMessage
from langchain.schema import HumanMessage

# Streamlit app title
st.title("💬 Groq Chatbot")

# Load Groq API key and model name from secrets
api_key = st.secrets["OPENAI_API_KEY"]
model_name = st.secrets["MODELNAME"]  # Example: "llama3-8b", "mixtral-8x7b"

# Initialize Groq Chat Model
llm = ChatGroq(
    temperature=0.7,  # Adjust for creativity
    model_name=model_name,
    groq_api_key=api_key
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box
user_input = st.chat_input("Type your message here...")

if user_input:    
    # Extract name
    name = extract_name(user_input)
    if name:
        st.session_state.user_name = name  # Store name in session state

    # Append user message to chat history    

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from Groq model
    response = llm.invoke([HumanMessage(content=user_input)])
    response_text = response.content

    # Append AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)