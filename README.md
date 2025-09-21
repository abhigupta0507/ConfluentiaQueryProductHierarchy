# Multilingual Query-Category Relevance

## Problem
Match multilingual search queries to product categories. Binary classification: relevant (1) or not (0).

## Solution
1. **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` for queries + categories
2. **Features**: Cosine similarity + embeddings + element product + abs difference  
3. **Classifier**: XGBoost

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage
1. Upload CSV with `origin_query` and `category_path` columns
2. Click "Run Inference" 
3. Download predictions

## Files
- `app.py` - Streamlit demo
- `model.py` - Model code (change file paths)
- `xgb_model.json` - Trained XGBoost model
- `requirements.txt` - Dependencies


# XGBoost Model Performance

## Overall F1-Score
**0.8253** (82.53%)

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0** (Not Relevant) | 0.67 | 0.30 | 0.41 | 674 |
| **1** (Relevant) | 0.74 | 0.93 | **0.83** | 1458 |

## Summary Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.73 |
| **Macro Avg** | 0.62 |
| **Weighted Avg** | 0.69 |
| **Total Samples** | 2132 |

## Key Insights
- Strong performance on positive class (F1: 0.83)
- High recall (0.93) for relevant queries
- Model slightly biased towards predicting relevant
  
