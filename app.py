import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title="TruthCheck AI", page_icon="📰")

st.title("📰 TruthCheck AI")
st.write("Detect whether a news statement is credible or misleading.")

text = st.text_area("Enter News Headline:", height=150)

if st.button("Analyze"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]

    positive = probs[1].item() * 100  # suggests real
    negative = probs[0].item() * 100  # suggests fake

    if positive > negative:
        st.success(f"✅ Likely Real ({positive:.2f}% confidence)")
    else:
        st.error(f"❌ Likely Fake ({negative:.2f}% confidence)")

    st.write("### Confidence Breakdown:")
    st.write(f"- Real: **{positive:.2f}%**")
    st.write(f"- Fake: **{negative:.2f}%**")
