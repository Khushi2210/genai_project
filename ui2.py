import streamlit as st
from app2 import (
    load_document, summarize_text,
    answer_question, generate_challenge_questions,
    evaluate_answer
)

st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("Smart Research Assistant")

uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded:
    ext = "." + uploaded.name.split(".")[-1].lower()

    if "text" not in st.session_state:
        with st.spinner("Processing document..."):
            st.session_state.text = load_document(uploaded, ext)
            st.session_state.summary = summarize_text(st.session_state.text)

    st.subheader("Auto Summary (≤150 words)")
    st.info(st.session_state.summary)

    mode = st.radio("Choose Interaction Mode:", ["Ask Anything", "Challenge Me"])

    # Ask Anything Mode
    if mode == "Ask Anything":
        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []

        question = st.text_input("Ask a question about the document:")

        if st.button("Get Answer"):
            with st.spinner("Analyzing..."):
                result = answer_question(st.session_state.text, question)
                st.session_state.qa_history.append((question, result))
                st.experimental_rerun()

        if st.session_state.qa_history:
            st.markdown("### Conversation History")
            for idx, (q, r) in enumerate(reversed(st.session_state.qa_history), 1):
                st.markdown(f"**Q{idx}:** {q}")
                st.markdown(f"**A{idx}:** {r['answer']}")
                with st.expander("Show Supporting Context"):
                    st.code(r["context"])
                st.markdown("---")

        if st.button("Clear Conversation"):
            st.session_state.qa_history = []
            st.experimental_rerun()

    # Challenge Me Mode
    elif mode == "Challenge Me":
        if "challenge_qs" not in st.session_state:
            with st.spinner("Generating questions..."):
                st.session_state.challenge_qs = generate_challenge_questions(st.session_state.text)
                st.session_state.challenge_evals = [None] * len(st.session_state.challenge_qs)

        for idx, qdata in enumerate(st.session_state.challenge_qs):
            st.markdown(f"**Q{idx+1}: {qdata['question']}**")
            user_input = st.text_input("Your Answer:", key=f"ans_{idx}")

            if st.button(f"Submit Answer {idx+1}", key=f"submit_{idx}"):
                st.session_state.challenge_evals[idx] = evaluate_answer(qdata, user_input)
                st.experimental_rerun()

            if st.session_state.challenge_evals[idx]:
                eval_result = st.session_state.challenge_evals[idx]
                feedback_str = str(eval_result["feedback"])
                if eval_result["is_correct"]:
                    st.success(feedback_str)
                else:
                    st.error(feedback_str)

                with st.expander("Supporting Text"):
                    st.write(eval_result["supporting_text"])
