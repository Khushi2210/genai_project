import os
import re
import tempfile
from typing import List, Dict
import streamlit as st
import nltk
#nltk.download('punkt')
#nltk.download('words')

from nltk.tokenize import sent_tokenize
from nltk.corpus import words
english_vocab = set(words.words())

import random
import fitz  # PyMuPDF
from transformers import pipeline
from summarizer import Summarizer
from sentence_transformers import SentenceTransformer, util

# ------------------ Model Loaders ------------------

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

# ------------------ Document Loader ------------------

def load_document(file_obj, extension: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_file.write(file_obj.read())
        temp_file.flush()
        temp_path = temp_file.name

    try:
        if extension == ".pdf":
            doc = fitz.open(temp_path)
            text = " ".join([page.get_text() for page in doc])
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
            pass  # Handle Windows file lock

    return re.sub(r'\s+', ' ', text).strip()

# ------------------ Summarization ------------------

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

# ------------------ BERT Embedding ------------------

def get_bert_embedding(text: str):
    return sentence_encoder.encode(text, convert_to_tensor=True)

# ------------------ Question Answering ------------------

def answer_question(text: str, question: str) -> Dict:
    try:
        # Preprocess the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences and create meaningful chunks
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        # Create chunks of 3-5 sentences each (better for context)
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) < 500:  # Word limit
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out very short or non-english chunks
        valid_chunks = []
        for chunk in chunks:
            words_in_chunk = chunk.split()
            if len(words_in_chunk) < 10:  # Skip very short chunks
                continue
                
            # Check if first few words are English (basic filter)
            first_words = " ".join(words_in_chunk[:3]).lower()
            if any(w in english_vocab for w in first_words.split()):
                valid_chunks.append(chunk)
        
        if not valid_chunks:
            valid_chunks = chunks  # fallback if no chunks passed filter
            
        # Get embeddings for question and chunks
        question_embed = get_bert_embedding(question)
        chunk_embeds = sentence_encoder.encode(valid_chunks, convert_to_tensor=True)
        
        # Calculate similarities and get top 3 most relevant chunks
        similarities = util.cos_sim(question_embed, chunk_embeds)[0]
        top_indices = similarities.topk(min(3, len(valid_chunks)))[1]
        top_chunks = [valid_chunks[i] for i in top_indices]
        
        # Combine top chunks for broader context
        context = " ".join(top_chunks)
        
        # Get answer from QA pipeline
        result = qa_pipeline(question=question, context=context, max_answer_len=150)
        
        # Find the exact sentence containing the answer for better justification
        answer_sentences = []
        for sent in sent_tokenize(context):
            if result["answer"].lower() in sent.lower():
                answer_sentences.append(sent)
                break
        
        justification = answer_sentences[0] if answer_sentences else top_chunks[0][:200] + "..."
        
        return {
            "answer": result["answer"].strip(),
            "context": justification,
            "confidence": round(float(result["score"]), 2),
            "full_context": context[:500] + "..." if len(context) > 500 else context
        }
        
    except Exception as e:
        return {
            "answer": f"Error processing your question: {str(e)}",
            "context": "",
            "confidence": 0.0,
            "full_context": ""
        }

# ------------------ Challenge Question Generator ------------------

def generate_challenge_questions(text: str, num_questions=3, min_chunk_size=300) -> List[Dict]:
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = sent_tokenize(text)

    if len(sentences) < num_questions * 3:
        return []

    chunks = []
    current_chunk = []
    word_count = 0

    for sent in sentences:
        current_chunk.append(sent)
        word_count += len(sent.split())

        if word_count >= min_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    questions = []

    for chunk in chunks[:num_questions]:
        chunk_sents = sent_tokenize(chunk)
        if len(chunk_sents) < 2:
            continue

        embeddings = sentence_encoder.encode(chunk_sents, convert_to_tensor=True)
        query = embeddings[0]
        sim_scores = util.cos_sim(query, embeddings)[0]

        top_indices = sim_scores.topk(min(3, len(chunk_sents)))[1].tolist()
        top_indices = [i for i in top_indices if i != 0]
        summary = ' '.join([chunk_sents[i] for i in top_indices[:2]]).strip()

        question_templates = [
            f"What is the main point of this passage: '{chunk_sents[0][:150]}...'?",
            f"Explain the key idea in this text: '{chunk_sents[0][:150]}...'",
            f"Summarize this content: '{chunk_sents[0][:150]}...'"
        ]
        question = random.choice(question_templates)

        questions.append({
            "question": question,
            "expected_answer": summary,
            "supporting_text": chunk
        })

    return questions

# ------------------ Evaluation ------------------

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
