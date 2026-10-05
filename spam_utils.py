"""
spam_utils.py
Shared preprocessing used by BOTH train.py and phishing_simulator.py.

Why a separate file? The saved model stores a reference to `clean_text`.
If that function lived inside train.py, loading the model from another
script would crash. Keeping it here means any script can import it.
"""
import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
PHONE_RE = re.compile(r"\b\d{5,}\b")          # long digit runs = phone / short codes
MONEY_RE = re.compile(r"[£$€]\s?\d+[\d,.]*")
NUM_RE = re.compile(r"\d+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Lowercase the message and replace noisy patterns with placeholder tokens.

    Spammers constantly change their exact URLs and phone numbers, so instead
    of learning '08452810075' we learn the general idea 'phonetoken'.
    """
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# Maps placeholder tokens back to something readable in the keyword report
TOKEN_LABELS = {
    "urltoken": "[a link/URL]",
    "emailtoken": "[an email address]",
    "moneytoken": "[a money amount]",
    "phonetoken": "[a phone number]",
    "numtoken": "[a number]",
}


def get_trigger_words(model, text: str, top_n: int = 8):
    """Return the words that pushed this message towards SPAM, strongest first.

    For Logistic Regression, each word's push = (its TF-IDF weight in this
    message) x (the coefficient the model learned for that word).
    Positive = pushes towards spam, negative = pushes towards ham.
    """
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    vec = vectorizer.transform([text])
    vocab = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]

    contributions = []
    for idx, tfidf_val in zip(vec.indices, vec.data):
        score = tfidf_val * coefs[idx]
        if score > 0:
            contributions.append((vocab[idx], float(score)))
    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions[:top_n]


def highlight(text: str, trigger_tokens, start="\033[1;41;97m", end="\033[0m"):
    """Wrap every original word whose cleaned form is a trigger token.

    Default style = bold white text on a red background (works in most
    terminals). Pass start="**", end="**" to get Markdown bold instead.
    """
    triggers = set(trigger_tokens)
    out = []
    for word in str(text).split(" "):
        cleaned = clean_text(word).split()
        if any(tok in triggers for tok in cleaned):
            out.append(f"{start}{word}{end}")
        else:
            out.append(word)
    return " ".join(out)
