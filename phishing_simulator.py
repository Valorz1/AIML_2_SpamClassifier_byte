"""
phishing_simulator.py — the "Phishing Simulator" twist for AVIP Task 2.

Run:  python phishing_simulator.py
Type or paste an email/message, then press Enter on an empty line to scan it.
Type 'quit' to exit.
"""
import os
import sys

import joblib

from spam_utils import TOKEN_LABELS, get_trigger_words, highlight

MODEL_PATH = "models/spam_model.joblib"

RED, GREEN, YELLOW, BOLD, RESET = "\033[91m", "\033[92m", "\033[93m", "\033[1m", "\033[0m"

# Make ANSI colours work in the Windows terminal too
if os.name == "nt":
    os.system("")


def risk_bar(prob: float, width: int = 30) -> str:
    filled = int(round(prob * width))
    colour = RED if prob >= 0.5 else (YELLOW if prob >= 0.25 else GREEN)
    return f"{colour}{'█' * filled}{RESET}{'░' * (width - filled)}"


def read_message() -> str:
    print(f"\n{BOLD}Paste/type your email below. Press Enter on an empty line to scan "
          f"(or type 'quit'):{RESET}")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().lower() == "quit" and not lines:
            return "quit"
        if line == "" and lines:
            break
        if line != "":
            lines.append(line)
    return " ".join(lines)


def scan(model, text: str):
    prob = float(model.predict_proba([text])[0][1])
    triggers = get_trigger_words(model, text, top_n=8)

    print("\n" + "=" * 60)
    print(f"{BOLD}  SPAM PROBABILITY SCORE: {prob * 100:5.1f}%{RESET}   {risk_bar(prob)}")
    if prob >= 0.5:
        print(f"  {RED}{BOLD}⚠  VERDICT: SPAM / POSSIBLE PHISHING{RESET}")
    elif prob >= 0.25:
        print(f"  {YELLOW}{BOLD}?  VERDICT: SUSPICIOUS — be careful{RESET}")
    else:
        print(f"  {GREEN}{BOLD}✓  VERDICT: LOOKS SAFE (HAM){RESET}")
    print("=" * 60)

    if triggers:
        print(f"\n{BOLD}Message with trigger keywords highlighted:{RESET}")
        print("  " + highlight(text, [w for w, _ in triggers]))
        print(f"\n{BOLD}Keywords that pushed the score towards SPAM:{RESET}")
        max_score = triggers[0][1]
        for word, score in triggers:
            label = TOKEN_LABELS.get(word, f"'{word}'")
            bar = "▇" * max(1, int(score / max_score * 15))
            print(f"  {label:<22} {RED}{bar}{RESET} {score:.3f}")
    else:
        print("\nNo spam-trigger keywords found.")


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found at {MODEL_PATH}. Run `python train.py` first.")
    model = joblib.load(MODEL_PATH)

    print(f"{BOLD}{'=' * 60}\n   B.Y.T.E PHISHING SIMULATOR  //  AVIP 2026 — Task 2\n{'=' * 60}{RESET}")
    while True:
        text = read_message()
        if text == "quit":
            print("Shutting down scanner. Stay safe.")
            break
        if text.strip():
            scan(model, text)


if __name__ == "__main__":
    main()
