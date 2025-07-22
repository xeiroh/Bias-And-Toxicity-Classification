

# ⚖️ Bias and Toxicity Classification using LSTM

## 📘 Overview

This project explores algorithmic bias in toxic comment classification using a deep learning model on the Jigsaw Unintended Bias in Toxicity Classification dataset. The project aims to understand and mitigate the unintended associations between identity terms (e.g., “gay”, “muslim”, “black”) and toxicity predictions.

---

## 🎯 Objective

To build a fair and effective text classification model that predicts comment toxicity, while analyzing the impact of sensitive identity terms and exploring fairness-aware modeling approaches.

---

## 📦 Dataset

- **Source**: Jigsaw / Kaggle / Civil Comments (via UC Berkeley’s Online Hate Index)
- **Size**: ~2 million comments
- **Features**:
  - `comment_text`: Raw text
  - `target`: Toxicity score (0–1)
  - Identity labels: `male`, `female`, `black`, `white`, `muslim`, `jewish`, etc.
  - Auxiliary labels: `insult`, `obscene`, `threat`, `identity_attack`, etc.
  - Reaction metadata: `likes`, `disagree`, `funny`, etc.

The dataset is highly imbalanced with a majority of non-toxic comments. Identity terms disproportionately appear in toxic contexts, risking model bias.

---

## 🧹 Preprocessing

- Lowercasing
- Spelling correction
- Null value handling
- FastText embedding tokenization

---

## 🧠 Model Architecture

Due to compute constraints, a moderately complex **BiLSTM** model was used:

```
Input → SpatialDropout → (128) BiLSTM → (128) BiLSTM → Mean/Max Pooling →
Dense(64) → Dense(512) → Dense(512) → Output
```

Custom loss functions were tested to penalize unfair subgroup performance and mitigate overfitting to biased patterns.

---

## 🧪 Evaluation Metrics

- **Final Score**: Weighted average AUC across subgroups
- **Subgroup AUCs**: Muslim, Black, White, Christian, etc.
- **BPSN/BNSP AUCs**: Fairness-oriented metrics for false positive/false negative sensitivity

---

## 📊 Results

- **Final AUC**: 0.9368 (vs. 0.94734 leaderboard best)
- High **BNSP AUC** shows good performance on toxic identity-linked comments
- Lower **BPSN AUC** indicates poor performance on non-toxic identity-linked comments

| Subgroup        | AUC    | BNSP   | BPSN   |
|-----------------|--------|--------|--------|
| gay/lesbian     | 0.847  | 0.957  | 0.881  |
| muslim          | 0.849  | 0.973  | 0.915  |
| jewish          | 0.876  | 0.938  | 0.934  |
| white           | 0.873  | 0.977  | 0.867  |
| female          | 0.935  | 0.970  | 0.944  |


<img width="485" height="486" alt="Screenshot 2025-07-21 at 9 19 20 PM" src="https://github.com/user-attachments/assets/b7133f4e-034b-46b0-b8a8-87f0331eb2d9" />
<img width="559" height="475" alt="Screenshot 2025-07-21 at 9 20 11 PM" src="https://github.com/user-attachments/assets/4b2fb803-fd23-4aec-bfa0-e73d3a081cf8" />

---

## 🔍 Model Interpretability

- **SHAP Summary Plot** revealed concerning feature importance for words like “black”, “white”, “trump”
- These terms correlated strongly with toxicity due to dataset imbalance, showing the model learns unintended associations
  
<img width="518" height="591" alt="Screenshot 2025-07-21 at 9 20 39 PM" src="https://github.com/user-attachments/assets/1e82b84a-f238-403b-b6d5-0b5818793cf1" />

---

## ✅ Conclusion

While the model achieved strong overall performance, analysis revealed **bias in decision-making**, especially for non-toxic comments referencing sensitive identities. SHAP interpretability highlighted the risks of unfair associations. Further research should focus on debiasing embeddings and expanding identity-balanced data.

---
