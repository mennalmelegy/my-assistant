import os
from groq import Groq

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM = (
    "Answer ONLY from the provided context. "
    "If the context does not contain the answer, say you don't know."
)


def call_llm(prompt: str) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return r.choices[0].message.content
