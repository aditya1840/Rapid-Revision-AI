import streamlit as st

from chatbot_core import ask_question, generate_notes

SUBJECTS = [
    "machine_learning",
    "deep_learning",
    "compiler_design",
    "data_structures",
    "environmental_science",
]

UNITS = ["UNIT-I", "UNIT-II", "UNIT-III", "UNIT-IV", "UNIT-V"]

st.set_page_config(
    page_title="Rapid Revision AI",
    layout="wide",
    page_icon="📘"
)

st.title("📘 Rapid Revision AI")
st.caption("AI-Powered Unit-Wise Exam Revision Assistant")

st.markdown(
    "Generate notes, revise concepts instantly, and prepare efficiently for university exams using Retrieval-Augmented Generation (RAG)."
)

subject = st.selectbox("Select Subject", SUBJECTS)
unit = st.selectbox("Select Unit", UNITS)

tab1, tab2 = st.tabs(["Ask Question", "Generate Notes"])

with tab1:
    question = st.text_input("Ask your question:")
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Generating answer..."):
                result = ask_question(subject=subject, unit=unit, question=question)
            st.subheader("Answer")
            st.write(result["answer"])
            if result["sources"]:
                st.caption("Sources: " + ", ".join(result["sources"]))

with tab2:
    if st.button("Generate Notes"):
        with st.spinner("Compiling notes..."):
            result = generate_notes(subject=subject, unit=unit)
        st.subheader("Unit Notes")
        st.write(result["answer"])
        if result["sources"]:
            st.caption("Sources: " + ", ".join(result["sources"]))
