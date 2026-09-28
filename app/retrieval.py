from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_chunks(data_dir: Path = DATA_DIR) -> list[dict]:
    chunks = []
    for f in sorted(data_dir.glob("*.md")):
        text = f.read_text(encoding="utf-8-sig")
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) > 30:
                chunks.append({"source": f.name, "text": para})
    return chunks


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in chunks])

    def search(self, query: str, k: int = 3) -> list[dict]:
        scores = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top if scores[i] > 0]
