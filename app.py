import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------- CONFIG ----------------------
MAX_LEN = 100
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# ---------------------- LOAD ARTIFACTS ----------------------
@st.cache_resource
def load_artifacts():
    model = load_model('model/best_model_v2.keras', compile=False)
    with open('model/tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('model/best_thresholds.pickle', 'rb') as f:
        thresholds = pickle.load(f)
    return model, tokenizer, thresholds

model, tokenizer, thresholds = load_artifacts()

# ---------------------- TEXT CLEANING (same as training) ----------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\[\[.*?\]\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_toxicity(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    proba = model.predict(padded, verbose=0)[0]
    result = {}
    for i, label in enumerate(LABEL_COLS):
        result[label] = {
            'probability': float(proba[i]),
            'is_toxic': bool(proba[i] >= thresholds[label])
        }
    return result

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="Comment Toxicity Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Comment Toxicity Detection")
st.write("Deep Learning (Bidirectional LSTM) based multi-label toxicity classifier trained on the Jigsaw dataset.")

tab1, tab2, tab3 = st.tabs(["🔍 Single Comment", "📁 Bulk Upload (CSV)", "📊 Model Insights"])

# ---------------- TAB 1: Single comment ----------------
with tab1:
    st.subheader("Enter a comment to analyze")
    user_input = st.text_area("Comment text", height=120, placeholder="Type or paste a comment here...")

    if st.button("Analyze Comment", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            result = predict_toxicity(user_input)
            any_toxic = any(v['is_toxic'] for v in result.values())

            if any_toxic:
                st.error("⚠️ This comment was flagged as potentially toxic.")
            else:
                st.success("✅ This comment appears clean.")

            st.write("### Detailed Breakdown")
            cols = st.columns(3)
            for i, (label, res) in enumerate(result.items()):
                with cols[i % 3]:
                    flag = "🔴" if res['is_toxic'] else "🟢"
                    st.metric(f"{flag} {label.replace('_', ' ').title()}", f"{res['probability']*100:.1f}%")

# ---------------- TAB 2: Bulk CSV upload ----------------
with tab2:
    st.subheader("Upload a CSV file for bulk predictions")
    st.caption("CSV must contain a column named 'comment_text'")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'comment_text' not in df.columns:
            st.error("CSV must contain a 'comment_text' column.")
        else:
            with st.spinner("Analyzing comments..."):
                predictions = []
                for text in df['comment_text']:
                    res = predict_toxicity(text)
                    row = {label: res[label]['is_toxic'] for label in LABEL_COLS}
                    predictions.append(row)
                pred_df = pd.DataFrame(predictions)
                output_df = pd.concat([df[['comment_text']], pred_df], axis=1)

            st.write(f"Processed {len(output_df)} comments")
            st.dataframe(output_df)

            csv_download = output_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv_download, "toxicity_predictions.csv", "text/csv")

# ---------------- TAB 3: Model Insights ----------------
with tab3:
    st.subheader("Model Performance & Insights")

    st.write("**Model Architecture:** Bidirectional LSTM with weighted binary cross-entropy loss")
    st.write("**Training Data:** 159,571 Wikipedia comments (Jigsaw Toxic Comment Classification dataset)")

    st.write("### Per-Label Decision Thresholds (F1-optimized)")
    thresh_df = pd.DataFrame({
        'Label': list(thresholds.keys()),
        'Threshold': [f"{v:.3f}" for v in thresholds.values()]
    })
    st.table(thresh_df)

    st.write("### Class Distribution in Training Data")
    class_dist = pd.DataFrame({
        'Label': LABEL_COLS,
        'Positive %': [9.58, 1.00, 5.29, 0.30, 4.94, 0.88]
    })
    st.bar_chart(class_dist.set_index('Label'))

    st.write("### Key Challenges Addressed")
    st.markdown("""
    - **Severe class imbalance**: 89.8% clean vs 10.2% toxic comments
    - **Rare classes**: 'threat' (0.3%) and 'identity_hate' (0.88%) required class-weighted loss
    - **Multi-label nature**: Used sigmoid + binary_crossentropy instead of softmax, since labels aren't mutually exclusive
    - **Threshold tuning**: Per-label F1-optimal thresholds instead of a universal 0.5 cutoff
    """)
    
    