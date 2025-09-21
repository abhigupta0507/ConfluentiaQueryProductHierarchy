import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb

train_csv = "/content/train.csv"
test_csv  = "/content/test.csv"



# Load data
train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)

# Load multilingual embedding model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Encode queries & categories
train_queries = model.encode(train_df["origin_query"].tolist(), show_progress_bar=True, batch_size=256)
train_cats    = model.encode(train_df["category_path"].tolist(), show_progress_bar=True, batch_size=256)

test_queries = model.encode(test_df["origin_query"].tolist(), show_progress_bar=True, batch_size=256)
test_cats    = model.encode(test_df["category_path"].tolist(), show_progress_bar=True, batch_size=256)


# Feature engineering
def make_features(q_emb, c_emb):
    # Calculate cosine similarity for each pair
    cos_sim = np.array([util.cos_sim(q, c).item() for q, c in zip(q_emb, c_emb)])

    # Calculate element-wise (Hadamard) product
    element_product = q_emb * c_emb

    # Calculate absolute difference
    abs_diff = np.abs(q_emb - c_emb)

    # Combine all features:
    # 1. Cosine Similarity (1 dimension)
    # 2. Raw Query Embedding (384 dimensions for this model)
    # 3. Raw Category Embedding (384 dimensions)
    # 4. Element-wise Product (384 dimensions)
    # 5. Absolute Difference (384 dimensions)
    feats = np.hstack([cos_sim.reshape(-1, 1), q_emb, c_emb, element_product, abs_diff])

    return feats

X = make_features(train_queries, train_cats)
y = train_df["label"].values

X_test = make_features(test_queries, test_cats)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# # Apply SMOTE to address class imbalance
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42, k_neighbors=5)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
X_train_resampled, y_train_resampled = X_train, y_train

# --- Try XGBoost (optional, usually better) ---
xgb_clf = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False
)
xgb_clf.fit(X_train_resampled, y_train_resampled)

val_preds_xgb = xgb_clf.predict(X_val)
print("XGBoost F1:", f1_score(y_val, val_preds_xgb, pos_label=1))
print(classification_report(y_val, val_preds_xgb))

# Final model = choose best (e.g. xgb_clf)
final_preds = xgb_clf.predict(X_test)

# Save submission
output_df = test_df.copy()
output_df["prediction"] = final_preds
output_path = "/content/test_predictions_fast.csv"
output_df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")


#   bst.update(dtrain, iteration=i, fobj=obj)
# XGBoost F1: 0.8252810695837132
#               precision    recall  f1-score   support

#            0       0.67      0.30      0.41       674
#            1       0.74      0.93      0.83      1458

#     accuracy                           0.73      2132
#    macro avg       0.70      0.61      0.62      2132
# weighted avg       0.72      0.73      0.69      2132

