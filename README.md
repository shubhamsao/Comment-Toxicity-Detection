# Comment Toxicity Detection using Deep Learning (BiLSTM)

A multi-label NLP classifier that detects toxic comments across 6 categories, deployed as an interactive Streamlit web app.

## Demo Video
🎥 [Watch on LinkedIn](https://www.linkedin.com/posts/shubham-sao-957453193_datascience-deeplearning-nlp-activity-7479474205319663616-flNX?utm_source=share&utm_medium=member_android&rcm=ACoAAC1zTpMBYkMfrdVYm0z5weKq8UYXsMBRjcU)

## Problem Statement
Online platforms need automated systems to detect toxic comments (harassment, hate speech, threats) in real-time to support content moderation. This project builds a deep learning model to classify comments across 6 toxicity categories simultaneously.

## Dataset
- **Source**: [Jigsaw Toxic Comment Classification Challenge (Kaggle)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- **Size**: 159,571 labeled Wikipedia comments
- **Labels**: toxic, severe_toxic, obscene, threat, insult, identity_hate (multi-label — a comment can belong to multiple categories)
- Full dataset hosted on Google Drive due to file size: [Drive Link](https://drive.google.com/drive/folders/19kmxnubw_tVyWEl5I7tQivc280MPvoTU?usp=sharing)

## Approach

### 1. Exploratory Data Analysis
- Confirmed no missing values or duplicates
- Identified severe class imbalance: 89.8% clean vs 10.2% toxic comments; rarest class `threat` at only 0.3%
- Analyzed comment length distribution (right-skewed) to determine sequence padding length

### 2. Text Preprocessing
- Lowercasing, removal of URLs, IP addresses, wiki-markup, emails, punctuation
- Tokenization (vocabulary size: 20,000) and padding (max length: 100 tokens)

### 3. Model Architecture
- Embedding layer (128 dimensions) → Bidirectional LSTM (64 units) → Dropout → Dense layers → Sigmoid output (6 independent binary classifications)
- **Loss function**: Custom weighted binary cross-entropy to address class imbalance — rare classes like `threat` penalized ~315x more heavily for misses during training
- **Evaluation**: AUC-ROC per label (0.95-0.99 across all classes) combined with per-label F1-optimized decision thresholds instead of a universal 0.5 cutoff

### 4. Deployment
- Streamlit web app with 3 features: single comment prediction, bulk CSV upload, model insights dashboard
- Model trained on Google Colab (T4 GPU) due to dataset size; inference runs locally on CPU

## Key Results
| Metric | Score |
|---|---|
| Mean AUC-ROC (macro) | 0.973 |
| Macro F1-score (tuned thresholds) | 0.53 |

## Tech Stack
Python, TensorFlow/Keras, Streamlit, Pandas, NumPy, Scikit-learn, Google Colab (GPU training)

## Project Structure
CommentToxicity/
├── app.py
├── requirements.txt
├── model/
│   ├── best_model_v2.keras
│   ├── tokenizer.pickle
│   └── best_thresholds.pickle
├── notebook/
│   └── Comment_Toxicity_Detection_Model_Training.ipynb
└── data/
└── sample_comments.csv

## How to Run Locally
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Challenges & Learnings
- Severe class imbalance required moving beyond accuracy to AUC-ROC and F1-score for honest evaluation
- Class-weighted loss improved recall on rare classes significantly but caused precision to drop (classic tradeoff), resolved via per-label threshold tuning
- Rarest class (`threat`, 0.3% of data) remains the weakest performer — a known limitation of LSTM approaches on extremely rare classes; transformer-based models (BERT) would likely perform better here with more compute budget

## Author
Shubham Sao