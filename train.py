"""
train.py — AVIP 2026 | AI/ML Task 2: Spam SMS / Email Classifier

Run:  python train.py
Produces:
  models/spam_model.joblib          trained pipeline (TF-IDF + Logistic Regression)
  outputs/metrics.json              accuracy / precision / recall / F1
  outputs/classification_report.txt per-class report
  outputs/confusion_matrix.png      confusion matrix on the test set
  outputs/top_spam_words.png        words the model treats as strongest spam signals
  outputs/example_predictions.csv   10 example inputs with labels + probabilities
"""
import json
import os
import urllib.request

import joblib
import matplotlib

matplotlib.use("Agg")  # save plots without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from spam_utils import clean_text, get_trigger_words

DATA_PATH = "data/sms.tsv"
# Mirror of the UCI SMS Spam Collection (tab-separated: label <TAB> message)
DATA_URL = ("https://raw.githubusercontent.com/justmarkham/"
            "pycon-2016-tutorial/master/data/sms.tsv")
RANDOM_STATE = 42


# ---------------------------------------------------------------- 1. Load data
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        print(f"Downloading dataset to {DATA_PATH} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    df = df.dropna().drop_duplicates(subset="message")   # remove repeated texts
    df["target"] = (df["label"] == "spam").astype(int)   # spam=1, ham=0
    return df


# ------------------------------------------------------------- 2. Build model
def build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,   # our custom cleaning from spam_utils.py
            stop_words="english",      # drop words like 'the', 'is', 'and'
            min_df=2,                  # ignore words that appear only once
            sublinear_tf=True,         # dampen words repeated many times
        )),
        ("clf", classifier),
    ])


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} unique messages "
          f"({df.target.sum()} spam / {(df.target == 0).sum()} ham)")

    # Stratify so the train and test sets keep the same spam/ham ratio
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["target"], test_size=0.2,
        stratify=df["target"], random_state=RANDOM_STATE)

    # ------------------------------------------------- 3. Train two models
    # Naive Bayes is the classic baseline; Logistic Regression is our main
    # model because its coefficients let us explain WHICH words caused a flag.
    baseline = build_pipeline(MultinomialNB()).fit(X_train, y_train)
    model = build_pipeline(LogisticRegression(
        C=10, class_weight="balanced", max_iter=2000,
        random_state=RANDOM_STATE)).fit(X_train, y_train)

    results = {
        "dataset": "UCI SMS Spam Collection",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "baseline_naive_bayes": evaluate(baseline, X_test, y_test),
        "logistic_regression": evaluate(model, X_test, y_test),
    }
    print(json.dumps(results, indent=2))
    with open("outputs/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=4)
    print(report)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    # ------------------------------------------------- 4. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Spam Classifier — Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()

    # ------------------------------------------------- 5. Top spam words chart
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    top = np.argsort(coefs)[-20:]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(vocab[top], coefs[top], color="#d62728")
    ax.set_xlabel("Model coefficient (higher = stronger spam signal)")
    ax.set_title("Top 20 Spam-Trigger Words Learned by the Model")
    plt.tight_layout()
    plt.savefig("outputs/top_spam_words.png", dpi=150)
    plt.close()

    # ------------------------------------------------- 6. 10 example predictions
    examples = [
        "URGENT! Your account has been suspended. Verify your details now at http://secure-bank-login.co to avoid closure.",
        "Congratulations! You have won a £1000 Walmart gift card. Call 09061701461 to claim your prize today!",
        "FREE entry to win an iPhone 17! Text WIN to 87121 now. T&Cs apply.",
        "Dear customer, you are selected for a cash reward of $500. Reply YES to claim before midnight.",
        "Limited time offer: get 50% off all subscriptions. Click here to claim your exclusive discount now!",
        "Hey, are we still meeting for lunch tomorrow at 1?",
        "Can you send me the notes from today's lecture when you get a chance?",
        "Hi Parmbir, the project review meeting has moved to Thursday at 3pm. See you there.",
        "Mum says dinner is ready, come home soon",
        "Thanks for the help with the assignment yesterday, really appreciated it!",
    ]
    rows = []
    for text in examples:
        prob = float(model.predict_proba([text])[0][1])
        triggers = [w for w, _ in get_trigger_words(model, text, top_n=5)]
        rows.append({
            "message": text,
            "predicted_label": "SPAM" if prob >= 0.5 else "HAM",
            "spam_probability": round(prob, 4),
            "top_trigger_words": ", ".join(triggers),
        })
    pd.DataFrame(rows).to_csv("outputs/example_predictions.csv", index=False)
    print(pd.DataFrame(rows)[["predicted_label", "spam_probability", "message"]]
          .to_string(max_colwidth=60))

    # ------------------------------------------------- 7. Save model
    joblib.dump(model, "models/spam_model.joblib")
    print("\nSaved model to models/spam_model.joblib")


if __name__ == "__main__":
    main()
