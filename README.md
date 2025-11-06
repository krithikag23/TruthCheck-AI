# 📰 TruthCheck AI — Fake News Detection Web App

TruthCheck AI is a **Fake News Detection** web application that uses a **pre-trained DistilBERT transformer model** to classify whether a news headline or short statement is **Likely Real** or **Likely Fake**, along with **confidence percentages**.  
This project runs **locally on any laptop (CPU only)** and requires **no model training** and **no dataset download** — everything works out of the box.

---

## ✨ Features
- 🔍 Analyzes headlines and short news statements
- 🤖 Powered by **DistilBERT** (Transformer-based NLP model)
- ⚡ Runs **fast** on CPU—no GPU required
- 🌐 Simple and elegant **Streamlit Web UI**
- 📊 Displays **confidence scores** for both Real and Fake
- 🔒 Fully offline — your text never leaves your system

---

## 🧠 How It Works
The app uses **DistilBERT sentiment analysis behavior**:
- **Positive / neutral tone → Likely Real**
- **Emotion-heavy / dramatic / manipulative tone → Likely Fake**

This approach works well because fake news often uses **fear, shock, exaggeration** and emotionally triggering wording.

---

## 🧪 Example Headlines to Try

| Prediction Type | Example Headline |
|-----------------|----------------|
| ✅ Likely Real | *"Electric vehicle adoption is increasing faster in rural regions than cities, new survey finds."* |
| ✅ Likely Real | *"WHO launches global program to strengthen pandemic detection systems."* |
| ❌ Likely Fake | *"Drinking lemon water daily eliminates cancer permanently, scientists confirm!"* |
| ❌ Likely Fake | *"Government has installed chips inside all new currency notes to track citizens."* |

Copy → Paste → Analyze inside the app.

---
