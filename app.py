import gradio as gr
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import matplotlib.pyplot as plt
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join(
            page.extract_text()
            for page in pdf.pages
            if page.extract_text()
        )

def analyze(job_desc, resumes):
    texts, names = [], []

    for r in resumes:
        texts.append(extract_text(r))
        names.append(r.name)

    job_emb = model.encode(job_desc, convert_to_tensor=True)
    res_embs = model.encode(texts, convert_to_tensor=True)

    scores = [util.cos_sim(job_emb, e).item() * 100 for e in res_embs]

    # 📊 BAR CHART
    fig_bar, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, scores)
    ax.set_ylabel("Match %")
    ax.set_title("Resume Match Comparison")
    plt.xticks(rotation=30)

    # 📈 GAUGE
    avg = np.mean(scores)
    fig_gauge, ax2 = plt.subplots(figsize=(6, 2))
    ax2.barh(["Average Match"], [avg])
    ax2.set_xlim(0, 100)
    ax2.set_title(f"Average Match Score: {avg:.2f}%")

    # 🧠 SKILL GAP
    jd_words = set(job_desc.lower().split())
    resume_words = set(" ".join(texts).lower().split())
    missing = ", ".join(list(jd_words - resume_words)[:20])

    ranking = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)

    return fig_gauge, fig_bar, ranking, missing


custom_css = """
body {
    background: #f5f7fb;
}
.gradio-container {
    max-width: 1200px !important;
}
"""

with gr.Blocks(css=custom_css, title="Semantic Resume Matcher") as demo:

    gr.Markdown("""
    # 🚀 Semantic Resume Matcher  
    ### AI-powered Resume Screening Dashboard  
    _Semantic · Context-aware · Recruiter-ready_
    """)

    with gr.Group():
        gr.Markdown("## 📥 Input Section")
        jd = gr.Textbox(label="📄 Job Description", lines=6, placeholder="Paste job description here...")
        resumes = gr.File(file_count="multiple", label="📂 Upload Resumes (PDF)")

        btn = gr.Button("🔍 Analyze Resumes", variant="primary")

    with gr.Group():
        gr.Markdown("## 📊 Analytics Overview")

        with gr.Row():
            gauge = gr.Plot(label="📈 Match Gauge")
            bar = gr.Plot(label="📊 Resume Comparison")

    with gr.Group():
        gr.Markdown("## 🏆 Ranking & Insights")

        table = gr.Dataframe(
            headers=["Resume", "Match Score (%)"],
            label="🏆 Resume Ranking",
            interactive=False
        )

        gap = gr.Textbox(
            label="🧠 Skill Gap (Missing Keywords)",
            lines=3
        )

    btn.click(analyze, [jd, resumes], [gauge, bar, table, gap])

demo.launch()
