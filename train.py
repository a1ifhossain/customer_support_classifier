"""Select a model using validation data, then evaluate it on held-out test data."""
import json
import os
import platform
from datetime import datetime, timezone
from uuid import uuid4

from config import INTENTS, ROOT, SEED

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from prepare_data import prepare, sha256


def build_pipeline(name):
    classifiers = {
        "logistic_regression": LogisticRegression(C=4.0, max_iter=1000, random_state=SEED),
        "naive_bayes": MultinomialNB(alpha=0.5),
    }
    # Fitting inside a Pipeline keeps vocabulary and IDF learning inside training.
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000)),
        ("classifier", classifiers[name]),
    ])


def scores(actual, predicted):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=sorted(INTENTS), average="macro", zero_division=0)),
    }


def main():
    reports = ROOT / "reports"
    artifacts = ROOT / "artifacts"
    reports.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    train, test, audit = prepare()
    fit, validation = train_test_split(train, test_size=0.2, stratify=train["category"], random_state=SEED)
    comparisons = []
    for name in ("logistic_regression", "naive_bayes"):
        candidate = build_pipeline(name)
        candidate.fit(fit["text"], fit["category"])
        result = {"model": name, **scores(validation["category"], candidate.predict(validation["text"]))}
        comparisons.append(result)
        print(f"Validation {name}: macro-F1={result['macro_f1']:.4f}", flush=True)
    # The test partition does not choose the model or hyperparameters.
    winner = max(comparisons, key=lambda result: result["macro_f1"])["model"]
    model = build_pipeline(winner)
    model.fit(train["text"], train["category"])
    prediction = model.predict(test["text"])
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train[["text"]], train["category"])
    baseline_prediction = baseline.predict(test[["text"]])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    metrics = {
        "run_id": run_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "Closed-set English intent classification on a selected 10-intent BANKING77 subset",
        "selected_intents": list(INTENTS),
        "selected_model": winner,
        "selection_metric": "Validation macro-F1",
        "validation_models": comparisons,
        "test_selected_model": scores(test["category"], prediction),
        "test_majority_baseline": scores(test["category"], baseline_prediction),
        "split_sizes": {"fit": len(fit), "validation": len(validation), "final_train": len(train), "test": len(test)},
        "seed": SEED,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "tfidf_parameters": {"ngram_range": [1, 2], "sublinear_tf": True, "max_features": 20000},
        "classifier_parameters": model.named_steps["classifier"].get_params(),
        "data_sha256": {split: sha256(ROOT / "data" / "raw" / f"{split}.csv") for split in ("train", "test")},
        "data_audit": audit,
    }
    bundle = {"pipeline": model, "intents": INTENTS, "metadata": metrics}
    joblib.dump(bundle, artifacts / "model.joblib")
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (reports / "run_history.jsonl").open("a", encoding="utf-8") as history:
        history.write(json.dumps(metrics) + "\n")
    pd.DataFrame(comparisons).to_csv(reports / "validation_comparison.csv", index=False)
    labels = sorted(INTENTS)
    report = classification_report(test["category"], prediction, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(reports / "classification_report.csv")
    matrix = confusion_matrix(test["category"], prediction, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(reports / "confusion_matrix.csv")
    predictions = test.rename(columns={"category": "actual"}).copy()
    predictions["predicted"] = prediction
    predictions["model_score_uncalibrated"] = model.predict_proba(test["text"]).max(axis=1)
    predictions.to_csv(reports / "test_predictions.csv", index=False)
    predictions.loc[predictions["actual"] != predictions["predicted"]].to_csv(reports / "errors.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 9))
    heatmap = ax.imshow(matrix, cmap="Blues")
    short_labels = [INTENTS[label] for label in labels]
    ax.set_xticks(range(len(labels)), short_labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), short_labels, fontsize=8)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    ax.set_title("BANKING77: selected 10 intents | held-out test confusion matrix")
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=8,
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    fig.colorbar(heatmap, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(reports / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    counts = train["category"].value_counts().reindex(labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(short_labels, counts.values, color="#287b8e")
    ax.set_xlabel("Training messages after cleaning")
    ax.set_title("Selected training data: class distribution")
    fig.tight_layout()
    fig.savefig(reports / "class_distribution.png", dpi=150)
    plt.close(fig)
    test_score = metrics["test_selected_model"]
    baseline_score = metrics["test_majority_baseline"]
    result_text = f"""# Evaluation results

Run: `{run_id}`  
Selected model: **{winner}**, chosen using validation macro-F1.  
Final training messages: **{len(train)}**. Held-out test messages: **{len(test)}**.

| Method | Test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| Majority-class baseline | {baseline_score['accuracy']:.4f} | {baseline_score['macro_f1']:.4f} |
| TF-IDF + {winner} | {test_score['accuracy']:.4f} | {test_score['macro_f1']:.4f} |

These results apply only to the selected ten English banking intents after the documented
duplicate audit. They are not a score on the complete 77-intent benchmark or evidence
of performance on live customer requests. See `errors.csv` for every incorrect prediction.

## Resume wording (after you can explain and reproduce the project)

Built a 10-intent banking support classifier using TF-IDF and {winner.replace('_', ' ')},
achieving {test_score['macro_f1']:.3f} macro-F1 on {len(test)} held-out messages after duplicate auditing;
compared two models using validation data and built a Streamlit prediction interface.

## Review before claiming completion

Run the app yourself, inspect at least five errors, and describe two limitations in your own words.
Training and demo code do not establish public deployment or production monitoring.
"""
    (reports / "RESULTS.md").write_text(result_text, encoding="utf-8")
    print(f"Winner: {winner}\nTest results: {test_score}\nBaseline: {baseline_score}", flush=True)
    print("Saved trained model, reports, figures, and run history.")


if __name__ == "__main__":
    main()
