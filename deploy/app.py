import streamlit as st
from inference import predict, OPTION_LABELS

st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠", layout="centered")

st.title("🧠 Smart MCQ Solver")
st.caption("Enter a question and its options to get the top-3 most likely answers.")

st.divider()

prompt = st.text_area("Enter Question", height=100, placeholder="Type the question here...")

st.subheader("Options")
option_values = {}
cols_labels = OPTION_LABELS  # ["A", "B", "C", "D", "E"]
for label in cols_labels:
    option_values[label] = st.text_input(f"Option {label}", key=f"option_{label}")

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("Please enter a question.")
    elif not any(v.strip() for v in option_values.values()):
        st.warning("Please enter at least one option.")
    else:
        filled_options = {k: v for k, v in option_values.items() if v.strip()}
        with st.spinner("Scoring options..."):
            results = predict(prompt, filled_options)

        st.subheader("🏆 Top 3 Predictions")

        medals = ["🥇", "🥈", "🥉"]
        for i, (label, prob) in enumerate(results[:3]):
            medal = medals[i] if i < len(medals) else "▫️"
            st.markdown(f"### {medal} {label} ({prob * 100:.1f}%)")

        st.divider()
        st.subheader("📊 All Ranked Options")
        table_data = {
            "Option": [label for label, _ in results],
            "Text": [filled_options[label] for label, _ in results],
            "Probability": [f"{prob * 100:.1f}%" for _, prob in results],
        }
        st.table(table_data)

st.divider()
st.caption("Smart MCQ Solver · BiLSTM model · Deployed with Streamlit Community Cloud")
