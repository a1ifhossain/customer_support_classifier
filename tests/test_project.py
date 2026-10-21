"""Checks for data leakage and real artifact/app behavior."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from config import ROOT
from predict import classify, load_model
from prepare_data import canonical_text, clean_split
from train import build_pipeline


def test_cleaning_removes_duplicate_and_conflicting_labels():
    raw = pd.DataFrame({
        "text": [" My card is late ", "my card is late", "ambiguous", "AMBIGUOUS", "", None],
        "category": ["card_arrival", "card_arrival", "change_pin", "card_arrival", "card_arrival", "card_arrival"],
    })
    clean, audit = clean_split(raw)
    assert clean["text"].tolist() == [" My card is late "]
    assert audit["duplicate_rows_removed"] == 1
    assert audit["conflicting_label_rows_removed"] == 2
    assert audit["empty_rows_removed"] == 2


def test_vectorizer_does_not_learn_unseen_words():
    pipeline = build_pipeline("logistic_regression")
    pipeline.fit(["missing card", "late card", "change pin", "new pin"], ["card_arrival", "card_arrival", "change_pin", "change_pin"])
    pipeline.predict(["uniquetesttoken"])
    assert "uniquetesttoken" not in pipeline.named_steps["tfidf"].vocabulary_


def test_prepared_partitions_have_no_exact_text_overlap():
    train = pd.read_csv(ROOT / "data" / "prepared" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "prepared" / "test.csv")
    assert set(train["text"].map(canonical_text)).isdisjoint(test["text"].map(canonical_text))


def test_saved_model_predicts_and_rejects_invalid_input():
    bundle = load_model()
    result = classify("My card was stolen yesterday", bundle)
    assert result[0]["intent"] == "lost_or_stolen_card"
    assert len(result) == 3
    assert 0 <= result[0]["score"] <= 1
    with pytest.raises(ValueError):
        classify("   ", bundle)
    with pytest.raises(ValueError):
        classify("a" * 2001, bundle)


def test_app_prediction_and_empty_input():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    app.button[0].click().run()
    assert app.warning
    app.text_area[0].set_value("My card was stolen yesterday")
    app.button[0].click().run()
    assert not app.exception
    assert app.success[0].value == "Lost or stolen card"
