import os
import re
import tempfile
from typing import List, Dict
import streamlit as st

import fitz  # PyMuPDF
from transformers import pipeline
from summarizer import Summarizer
from sentence_transformers import SentenceTransformer, util

# Model Loaders with caching
@st.cache_resource
def load_summarizer():
    return Summarizer()

@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@st.cache_resource
def load_sentence_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")

summarizer = load_summarizer()
qa_pipeline = load_qa_pipeline()
sentence_encoder = load_sentence_encoder()

# Load document using PyMuPDF
def load_document(file_obj, extension: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_file.write(file_obj.read())
        temp_file.flush()
        temp_path = temp_file.name

    try:
        if extension == ".pdf":
            doc = None
            try:
                doc = fitz.open(temp_path)
                text = " ".join([page.get_text() for page in doc])
            finally:
                if doc:
                    doc.close()
        elif extension == ".txt":
            with open(temp_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            raise ValueError("Unsupported file type.")
    finally:
        try:
            os.remove(temp_path)
        except PermissionError:
            pass  # File still in use on Windows — skip deletion

    return re.sub(r'\s+', ' ', text).strip()

# Summarization
def summarize_text(text: str, chunk_size=1000, max_chunks=3) -> str:
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks[:max_chunks]:
        try:
            result = summarizer(chunk, min_length=60, max_length=150)
            summaries.append(result)
        except Exception as e:
            summaries.append(f"(Error summarizing chunk: {e})")
    return " ".join(summaries)

# BERT Embedding
def get_bert_embedding(text: str):
    return sentence_encoder.encode(text, convert_to_tensor=True)

# Question Answering + Snippet Retrieval
def answer_question(text: str, question: str) -> Dict:
    try:
        chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
        question_embed = get_bert_embedding(question)
        similarities = [util.cos_sim(get_bert_embedding(chunk), question_embed).item() for chunk in chunks]
        best_chunk = chunks[similarities.index(max(similarities))]
        result = qa_pipeline(question=question, context=best_chunk)
        return {
            "answer": result["answer"],
            "context": best_chunk.strip()
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "context": ""
        }

# Challenge Question Generator

def generate_challenge_questions(text: str, skip_words: int = 200) -> List[Dict]:
    """
    Skip first N words (default 300) to avoid author/abstract info, then
    generate 3 meaningful logic/comprehension questions using BERT-based method.
    """
    # 1. Skip the first few words
    words = text.split()
    text_main = " ".join(words[skip_words:]) if len(words) > skip_words else text

    # 2. Split into chunks
    chunks = [text_main[i:i + 1000] for i in range(0, len(text_main), 1000)]
    questions = []

    for i, chunk in enumerate(chunks):
        sentences = re.split(r'(?<=[.!?]) +', chunk.strip())
        if not sentences:
            continue
        main_sentence = max(sentences, key=len)  # Use longest sentence as base for summary

        question = f"Q{i+1}: What does the following statement mean? '{main_sentence.strip()}'"
        questions.append({
            "question": question,
            "expected_answer": summarize_text(main_sentence),  # generate semantic summary
            "supporting_text": chunk.strip()
        })

        if len(questions) == 3:
            break

    return questions



# Evaluate Challenge Answers with BERT Similarity
def evaluate_answer(question_data: Dict, user_answer: str) -> Dict:
    expected = question_data["expected_answer"]
    supporting_text = question_data["supporting_text"]

    expected_embed = get_bert_embedding(expected)
    user_embed = get_bert_embedding(user_answer)
    similarity = util.cos_sim(expected_embed, user_embed).item()

    if similarity > 0.75:
        feedback = f"Your answer is semantically correct. (Similarity: {similarity:.2f})"
        is_correct = True
    elif similarity > 0.60:
        feedback = f"Your answer is somewhat close but not fully correct. (Similarity: {similarity:.2f})"
        is_correct = False
    else:
        feedback = f"Your answer is incorrect. Expected something like: '{expected}'. (Similarity: {similarity:.2f})"
        is_correct = False

    return {
        "is_correct": is_correct,
        "feedback": feedback,
        "supporting_text": supporting_text
    }
