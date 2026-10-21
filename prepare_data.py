"""Download public BANKING77 CSVs, select ten intents, and audit duplicates."""
import hashlib
import json
import re
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import pandas as pd

from config import INTENTS, ROOT

SOURCE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"


def canonical_text(text):
    """Only for auditing duplicates; the ML pipeline receives the original text."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download():
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "source_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries = {}
    for split in ("train", "test"):
        path = raw_dir / f"{split}.csv"
        url = f"{SOURCE}/{split}.csv"
        if not path.exists():
            print(f"Downloading official {split}.csv ...", flush=True)
            request = Request(url, headers={"User-Agent": "BANKING77-learning-project"})
            with urlopen(request, timeout=60) as response:
                contents = response.read()
            # Check the response before saving, including accidentally downloaded HTML.
            if not contents.startswith(b"text,category"):
                raise ValueError(f"Unexpected dataset format at {url}")
            path.write_bytes(contents)
        digest = sha256(path)
        old = previous.get("files", {}).get(split, {}).get("sha256")
        if old and digest != old:
            raise ValueError(f"{path} differs from the recorded source snapshot.")
        entries[split] = {"url": url, "sha256": digest, "bytes": path.stat().st_size}
    manifest = {
        "dataset": "BANKING77 (PolyAI)",
        "license": "CC BY 4.0",
        "repository": "https://github.com/PolyAI-LDN/task-specific-datasets",
        "first_download_utc": previous.get("first_download_utc", datetime.now(timezone.utc).isoformat()),
        "files": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def clean_split(frame):
    """Apply a fixed audit policy independently to each official partition."""
    if not {"text", "category"}.issubset(frame.columns):
        raise ValueError("Expected columns named text and category.")
    audit = {"source_rows": len(frame)}
    frame = frame.loc[frame["category"].isin(INTENTS), ["text", "category"]].copy()
    audit["selected_rows"] = len(frame)
    valid = frame["text"].notna() & frame["text"].astype(str).str.strip().ne("")
    audit["empty_rows_removed"] = int((~valid).sum())
    frame = frame.loc[valid].copy()
    frame["_key"] = frame["text"].map(canonical_text)
    counts = frame.groupby("_key")["category"].nunique()
    ambiguous = frame["_key"].isin(counts[counts > 1].index)
    audit["conflicting_label_rows_removed"] = int(ambiguous.sum())
    frame = frame.loc[~ambiguous].copy()
    before = len(frame)
    frame = frame.drop_duplicates("_key")
    audit["duplicate_rows_removed"] = before - len(frame)
    return frame.reset_index(drop=True), audit


def prepare():
    download()
    raw_dir = ROOT / "data" / "raw"
    train, train_audit = clean_split(pd.read_csv(raw_dir / "train.csv"))
    test, test_audit = clean_split(pd.read_csv(raw_dir / "test.csv"))
    # Never move official test examples into training. Remove overlaps from test.
    overlaps = test["_key"].isin(set(train["_key"]))
    test_audit["train_overlap_rows_removed"] = int(overlaps.sum())
    test = test.loc[~overlaps].reset_index(drop=True)
    for name, frame in (("train", train), ("test", test)):
        if set(frame["category"]) != set(INTENTS):
            raise ValueError(f"Missing selected categories in {name} partition.")
    prepared = ROOT / "data" / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    train.drop(columns="_key").to_csv(prepared / "train.csv", index=False)
    test.drop(columns="_key").to_csv(prepared / "test.csv", index=False)
    audit = {"train": train_audit, "test": test_audit, "final_train_rows": len(train), "final_test_rows": len(test)}
    (prepared / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return train.drop(columns="_key"), test.drop(columns="_key"), audit


if __name__ == "__main__":
    _, _, summary = prepare()
    print(json.dumps(summary, indent=2))
