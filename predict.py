"""Load our own trained artifact and predict a support intent."""
import argparse

import joblib

from config import ROOT


def load_model():
    path = ROOT / "artifacts" / "model.joblib"
    if not path.exists():
        raise FileNotFoundError("Train the model first: python train.py")
    return joblib.load(path)


def classify(text, bundle):
    text = text.strip()
    if not text:
        raise ValueError("Please enter a message.")
    if len(text) > 2000:
        raise ValueError("Use a message of at most 2,000 characters.")
    pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    order = probabilities.argsort()[::-1][:3]
    return [
        {"intent": str(classes[index]), "label": bundle["intents"][classes[index]], "score": float(probabilities[index])}
        for index in order
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", help="A short English banking support message")
    args = parser.parse_args()
    try:
        for rank, item in enumerate(classify(args.message, load_model()), start=1):
            print(f"{rank}. {item['label']} ({item['intent']}): {item['score']:.3f}")
        print("Scores are uncalibrated; the model always chooses among the ten supported intents.")
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, str(error) + "\n")
