import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "mohamadsoltani/Fake-News-BERT-Base-uncased"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title="TruthCheck AI", page_icon="📰", layout="centered")

st.title("📰 TruthCheck AI")
st.write("Enter a news headline or statement below:")

text = st.text_area("News Text", height=150)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
            fake_prob = probs[0].item() * 100
            real_prob = probs[1].item() * 100

        st.write("### Result:")
        if fake_prob > real_prob:
            st.error(f"❌ Fake News Detected ({fake_prob:.2f}% confidence)")
        else:
            st.success(f"✅ Real News Likely ({real_prob:.2f}% confidence)")

        st.write("### Confidence Breakdown:")
        st.write(f"- Fake: **{fake_prob:.2f}%**")
        st.write(f"- Real: **{real_prob:.2f}%**")
