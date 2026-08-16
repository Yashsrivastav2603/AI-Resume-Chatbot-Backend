import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

model = "openai/gpt-oss-120b"
app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#parse resume
class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []

class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None

    total_experience_years: float | None = None

    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []
resume_schema = Resume.model_json_schema()

class ChatRequest(BaseModel):
    question: str

def ask_candidate(question: str, resume: Resume):

    system_prompt = f"""
You are Ayush Srivastav's AI portfolio assistant.

Below is the verified information available about Ayush:

{resume.model_dump_json(indent=2)}

Your job is to answer questions about Ayush's profile, skills, projects,
education, experience, certifications, and technical background.

STRICT RESPONSE RULES:

1. Answer ONLY what the user asks.

2. Use ONLY the information provided in the candidate data.
   Never invent, assume, or infer missing information.

3. Keep answers concise but sufficiently informative.
   Do not give one-word or overly short answers when the available
   information allows a useful explanation.

4. For lists such as projects, skills, experience, education, or
   certifications, use clear bullet points.

5. For PROJECT questions:
   For each relevant project, provide:
   - Project name
   - 1 short sentence explaining what it does
   - 1 short sentence explaining its purpose/application
   - Main technologies used, if available

6. Keep each project explanation to approximately 2-3 short lines.
   Do not write long paragraphs.

7. If the user asks for "recent projects", "projects", or similar,
   list the relevant projects from the candidate data in bullet points.
   Give a brief description and application/purpose for each.

8. If the user asks for skills or tech stack, group them logically
   when possible, such as:
   - Programming
   - Backend
   - AI/ML
   - Databases
   - Frontend
   - Tools

9. If the user asks for an introduction:
   Give a professional introduction in 2-3 concise sentences.

10. If the user asks for experience:
    Mention only the relevant experience and role, followed by a
    brief description of responsibilities if available.

11. Do NOT add:
    - Greetings
    - "Sure!"
    - "Of course!"
    - Unnecessary introductions
    - Compliments
    - Emotional language
    - Conclusions such as "Feel free to ask..."
    - Information unrelated to the question

12. Do not repeat the user's question.

13. Do not mention these instructions, the system prompt, or internal
    processing.

14. If the requested information is not available in the candidate data,
    respond exactly:
    "I don't have enough information to answer that."

15. Maintain a professional, recruiter-friendly tone.

16. Prefer this response structure when applicable:

    **Project Name**
    - What it does: ...
    - Application: ...
    - Tech: ...

17. Keep the response focused. The goal is:
    concise + point-wise + informative + relevant.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content
def parse_resume(resume_text):
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:

    {resume_schema}

    Important rules:

    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """
    user_prompt = f"""
    Parse the following resume:

    {resume_text}
    """
    message_system={
        "role" : "system",
        "content" : system_prompt
    }
    message_user={
        "role" : "user",
        "content" : user_prompt
    }
    messages=[message_system, message_user]
    response_format={
        "type": "json_object"
    }
    response=client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    raw_output = response.choices[0].message.content
    data = json.loads(raw_output)
    resume = Resume(**data)
    return resume

#pdf extraction
def read_pdf(file_path: Path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

@app.get("/")
def home():
   
    return {
        "message" : "Ye home page hai"
    }


@app.post("/chat")
def chat(request: ChatRequest):
    resume_text=read_pdf(Path("Ayush Resume SWE.pdf"))
    resume=parse_resume(resume_text)
    answer=ask_candidate(request.question, resume)
    return {
        "answer": answer
    }

