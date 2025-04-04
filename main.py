import os
import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import google.generativeai as genai
import PyPDF2 as pdf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Generative AI API
genai.configure(api_key=("AIzaSyC-qSvpq44LP0hYgr7EZLZaOIxlPzezP3g"))

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware


# Enable CORS for your frontend (Replace with your actual frontend URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend origin
    allow_credentials=True,  # Allow cookies, authentication
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)



# Convert PDF to text
def extract_text_from_pdf(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Get response from Generative AI
def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.candidates[0].content.parts[0].text if response else None

@app.post("/generate-fitment-reports/")
async def generate_fitment_reports(resumes: List[UploadFile] = File(...), job_description: UploadFile = File(...)):
    try:
        job_description_text = extract_text_from_pdf(job_description.file)

        fitment_results = []

        for resume in resumes:
            candidate_name = os.path.splitext(resume.filename)[0]
            resume_text = extract_text_from_pdf(resume.file)

            # Construct the prompt (company culture content removed)
            input_prompt = f"""
### Task: Generate a candidate shortlisting report.
### Instructions:
You are a highly intelligent and unbiased system designed to shortlist candidates for a job based on:
1. The candidate's resume.
2. A provided job description.
### Key Objectives:
- Accurate Matching the Skills from job description and resumes.
- Analyze skills, qualifications, and experiences in the resume.
- Evaluate alignment with the job description.
- Provide detailed scoring, strengths, weaknesses, and recommendations.
### Required Sections in the Report:
- Candidate Name and Email
- Parse properly All the job description and create a 'Should Do' list, categorizing required skills into levels: Beginner, Competent, Intermediate, Expert by Studying and analysing job title, there requirements and all.
- Parse properly All the candidate's resume and create a 'Can Do' list, categorizing listed skills into the same levels: Beginner, Competent, Intermediate, Expert. To categorize the skill see whether there are certificates, projects, internship experinece, any other experinece.
- Matching score: Match the created 'can do' and 'should do' list. To generate the matching Score use strategy as if skill level from both list is same then give it 100 and decrease 25 for each difference in skill levels from should do and can do list.And if can do skill level is greater than should do skill level then give 100.To calculate final Matching score make average of the sum of all can do skill scores. 
- Analysis of strengths and weaknesses.
- Recommendations for improvement.
- Overall conclusion.
### Input Data:
- **Resume**: {resume_text}
- **Job Description**: {job_description_text}
### Output Format:
1. Candidate Name 
2. Email
3. "Can Do" list:
4. "Should Do" list
5. Skill Comparison Table:
   | Skill                   | "Can Do" Level  | "Should Do" Level  | Matching Score |
   |--------------------------|----------------|--------------------|----------------|
6. Overall Matching Score: [Percentage]
7. Analysis of Strengths and Weaknesses
8. Recommendations for Improvement
9. Conclusion on Fitment
Generate Accurate Report of the candidate.
Note: Remove or do not generate the words 'Ok','Okay' and the sentence like 'Okay, I will generate a candidate shortlisting report for' from the generated PDF of the fitment report.
            """

            report_content = get_gemini_response(input_prompt)

            import re

            if report_content:
                try:
                    # Use regex to extract the matching score
                    match = re.search(r"Overall Matching Score:\s*([\d.]+)%", report_content)
                    if match:
                        matching_score = float(match.group(1))  # Extract numeric value
                    else:
                        matching_score = 0.0
                        report_content += "\n\n[ERROR: Matching Score could not be parsed]"
                except ValueError:
                    matching_score = 0.0
                    report_content += "\n\n[ERROR: Matching Score could not be parsed]"

                fitment_results.append({
                    "candidate_name": candidate_name,
                    "matching_score": matching_score,
                    "report": report_content
                })

        # Sort candidates by matching score in descending order
        fitment_results.sort(key=lambda x: x["matching_score"], reverse=True)

        return {"Candidates": fitment_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating fitment reports: {str(e)}") 

# Ensure Render uses the correct port
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Get port from environment
    uvicorn.run(app, host="0.0.0.0", port=port)









# Best version with CV RAG

# import os
# import json
# from datetime import datetime
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from typing import List
# import google.generativeai as genai
# import PyPDF2 as pdf
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

# # Load environment variables
# load_dotenv()

# # Configure Generative AI API
# genai.configure(api_key=("AIzaSyC-qSvpq44LP0hYgr7EZLZaOIxlPzezP3g"))

# app = FastAPI()

# # Initialize vectorstore
# def setup_vectorstore():
#     embeddings = HuggingFaceEmbeddings()
#     vectorstore = Chroma(persist_directory="cv_vectordb", embedding_function=embeddings)
#     return vectorstore

# # vectorstore = setup_vectorstore()

# # Convert PDF to text
# def extract_text_from_pdf(uploaded_file):
#     reader = pdf.PdfReader(uploaded_file)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text() or ""
#     return text

# # Retrieve relevant content from vectorstore
# # def retrieve_from_vectorstore(query):
# #     retriever = vectorstore.as_retriever()
# #     results = retriever.invoke(query)
# #     return "\n".join([doc.page_content for doc in results])

# # Lazy Load VectorStore inside the function
# def retrieve_from_vectorstore(query):
#     embeddings = HuggingFaceEmbeddings()
#     vectorstore = Chroma(persist_directory="cv_vectordb", embedding_function=embeddings)  # Load only when needed
#     retriever = vectorstore.as_retriever()
#     results = retriever.invoke(query)
#     return "\n".join([doc.page_content for doc in results])


# # Get response from Generative AI
# def get_gemini_response(prompt):
#     model = genai.GenerativeModel('gemini-2.0-flash')
#     response = model.generate_content(prompt)
#     return response.candidates[0].content.parts[0].text if response else None

# @app.post("/generate-fitment-reports/")
# async def generate_fitment_reports(resumes: List[UploadFile] = File(...), job_description: UploadFile = File(...)):
#     try:
#         job_description_text = extract_text_from_pdf(job_description.file)
#         company_culture_content = retrieve_from_vectorstore("company culture match")

#         fitment_results = []

#         for resume in resumes:
#             candidate_name = os.path.splitext(resume.filename)[0]
#             resume_text = extract_text_from_pdf(resume.file)

#             # Construct the prompt (unchanged)
#             input_prompt = f"""
# ### Task: Generate a candidate shortlisting report.
# ### Instructions:
# You are a highly intelligent and unbiased system designed to shortlist candidates for a job based on:
# 1. The candidate's resume.
# 2. A provided job description.
# 3. Relevant company culture data retrieved from the vector database.
# ### Key Objectives:
# - Accurate Matching the Skills from job description and resumes.
# - Analyze skills, qualifications, and experiences in the resume.
# - Evaluate alignment with the job description.
# - Assess cultural fit using company culture data.
# - Provide detailed scoring, strengths, weaknesses, and recommendations.
# ### Required Sections in the Report:
# - Candidate Name and Email
# - Parse properly All the job description and create a 'Should Do' list, categorizing required skills into levels: Beginner, Competent, Intermediate, Expert by Studying and analysing job title, there requirements and all.
# - Parse properly All the candidate's resume and create a 'Can Do' list, categorizing listed skills into the same levels: Beginner, Competent, Intermediate, Expert. To categorize the skill see whether there are certificates, projects, internship experinece, any other experinece.
# - Matching score: Match the created 'can do' and 'should do' list. To generate the matchinging Score use strategy as if skill level from both list is same then give it 100 and decrease 25 for each difference in skill levels from should do and can do list.And if can do skill level is greater than should do skill level then give 100.To calculate final Mathching score make the of all can do skill score.
# - Analysis of strengths and weaknesses.
# - Recommendations for improvement.
# - Overall conclusion.
# ### Input Data:
# - **Resume**: {resume_text}
# - **Job Description**: {job_description_text}
# - **Company Culture Data**: {company_culture_content}
# ### Output Format:
# 1. Candidate Name and Email
# 2."Can Do" list:
# 3. "Should Do" list
# 4. Skill Comparison Table:
#    | Skill                   | "Can Do" Level  | "Should Do" Level  | Matching Score |
#    |--------------------------|----------------|--------------------|----------------|
# 5. Overall Matching Score: [Percentage]
# 6. Analysis of Strengths and Weaknesses
# 7. Recommendations for Improvement
# 8. Conclusion on Fitment
# Generate Accurate Report of the candedate.
# Note:Remove or do not generate the words 'Ok','Okay'and the sentence like 'Okay, I will generate a candidate shortlisting report for ' from the generated pdf of the  fitment report
#             """

#             report_content = get_gemini_response(input_prompt)

#             import re

#             if report_content:
#                 try:
#                     # Use regex to extract the matching score
#                     match = re.search(r"Overall Matching Score:\s*([\d.]+)%", report_content)
#                     if match:
#                         matching_score = float(match.group(1))  # Extract numeric value
#                     else:
#                         matching_score = 0.0
#                         report_content += "\n\n[ERROR: Matching Score could not be parsed]"
#                 except ValueError:
#                     matching_score = 0.0
#                     report_content += "\n\n[ERROR: Matching Score could not be parsed]"

#                 fitment_results.append({
#                     "candidate_name": candidate_name,
#                     "matching_score": matching_score,
#                     "report": report_content
#                 })


#         # Sort candidates by matching score in descending order
#         fitment_results.sort(key=lambda x: x["matching_score"], reverse=True)

#         return {"Candidates": fitment_results}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating fitment reports: {str(e)}")
