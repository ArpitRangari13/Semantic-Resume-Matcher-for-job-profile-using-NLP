import gradio as gr
from sentence_transformers import SentenceTransformer, util
import pdfplumber

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join(
            page.extract_text() for page in pdf.pages if page.extract_text()
        )

def match_resume(job_desc, resume_pdf):
    resume_text = extract_text(resume_pdf)
    emb_job = model.encode(job_desc, convert_to_tensor=True)
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    score = util.cos_sim(emb_job, emb_resume).item()
    return f"🎯 Match Score: {round(score * 100, 2)} %"

with gr.Blocks(title="Semantic Resume Matcher Dashboard") as demo:
    gr.Markdown("## 🚀 Semantic Resume Matcher")
    gr.Markdown("AI-powered resume screening dashboard")

    with gr.Row():
        job_desc = gr.Textbox(label="Job Description", lines=8)
        resume = gr.File(label="Upload Resume (PDF)")

    btn = gr.Button("🔍 Analyze Resume")
    output = gr.Textbox(label="Result")

    btn.click(match_resume, [job_desc, resume], output)

demo.launch()
