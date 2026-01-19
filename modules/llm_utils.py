from langchain_ollama import OllamaLLM
from PIL import Image
import io

def answer_from_own_knowledge(question: str, text_model: str) -> str:
    llm = OllamaLLM(model=text_model)
    prompt = f"""
Answer ONLY from your own knowledge.
If unsure, reply EXACTLY:
I DON'T KNOW

Question:
{question}

Answer:
"""
    return llm.invoke(prompt).strip()

def answer_from_document(question: str, context: str, text_model: str) -> str:
    llm = OllamaLLM(model=text_model)
    prompt = f"""
You MUST answer ONLY using the context below.
If the answer is not in the context, say EXACTLY:
ANSWER NOT IN DOCUMENT

Context:
{context}

Question:
{question}

Answer:
"""
    return llm.invoke(prompt).strip()

def image_answer(image_file, question: str) -> str:
    llm = OllamaLLM(model="llava")
    img = Image.open(image_file)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return llm.invoke(question, images=[buf.getvalue()])
