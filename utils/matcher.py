from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocess import preprocess

model = SentenceTransformer("all-MiniLM-L6-v2")

def match_resume(resume_text, jd_text):
    resume_clean = preprocess(resume_text)
    jd_clean = preprocess(jd_text)

    resume_emb = model.encode(resume_clean)
    jd_emb = model.encode(jd_clean)

    score = cosine_similarity([resume_emb], [jd_emb])[0][0]
    return round(score * 100, 2)