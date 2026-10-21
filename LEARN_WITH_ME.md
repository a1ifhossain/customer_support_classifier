# Learn with the project

## Lesson 1: Features and labels

A training example has an input and the answer we want a model to learn.

```python
example = {
    "text": "I lost my card yesterday",
    "category": "lost_or_stolen_card",
}
print(example["text"])
print(example["category"])
```

This is a Python **dictionary**: a collection of keys and values. `text` is our input;
`category` is its known label. A list can hold many examples. A pandas DataFrame stores
the examples as a table, with one row per message and one column per field.

Exercise: create two more dictionaries for a missing card delivery and a forgotten passcode.
Write the difference between those inputs and their labels in `MY_NOTES.md`.

## Lesson 2: Read and inspect the data

From the project terminal, enter the interactive Python shell:

```powershell
& '.\.venv\Scripts\python.exe'
```

Then type:

```python
import pandas as pd
data = pd.read_csv("data/prepared/train.csv")
print(data.head())
print(data.shape)
print(data["category"].value_counts())
```

`import` loads a library. `pd` is a short alias for pandas. `read_csv` loads a table.
`head()` shows the first rows; `shape` gives row and column counts.
`value_counts()` counts occurrences of each label.

Exercise: find the least common category. How much larger is the biggest category?
Use `exit()` to leave the Python shell.

## Lesson 3: Why clean the data?

In `prepare_data.py`, we remove empty messages, exact duplicates after lowercasing and
whitespace normalization, and messages with conflicting labels within each partition.
We remove exact normalized test/train overlaps from the test partition. We keep the
official training and test partitions separate throughout.

This reduces a simple form of data leakage: being tested on the same message used
for training. Similar paraphrases can still remain; exact deduplication is not a
semantic similarity audit. Cleaning rules were set before examining model results.

Exercise: compare `"My card is late"` and `"  my CARD is late  "` using `canonical_text`.
Read `data/prepared/data_audit.json` to see what was actually removed.

## Lesson 4: TF-IDF

The classifier needs numbers, so TF-IDF converts each message into a sparse numerical
vector. It gives weight to terms based on their frequency in a document and across the
training corpus. It is a statistical text representation, not language understanding.

Our vectorizer uses single words and adjacent pairs of words. The pair `"change pin"`
can help distinguish that request from a forgotten app passcode.

Open `train.py` and find `build_pipeline`. The `Pipeline` connects vectorization and
classification. Its vocabulary and inverse document frequencies are learned only
when we fit on training examples. Test data is transformed using that fitted state.

Exercise: name two word pairs that might help recognize a request to close an account.

## Lesson 5: Training and model selection

The project compares two predetermined choices:

- **Logistic Regression:** learns a weighted relationship between text features and each class.
- **Multinomial Naive Bayes:** uses a probabilistic model of feature/class relationships.

Both have assumptions and can make mistakes. We do not choose by which name sounds
more advanced. The winner is chosen using macro-F1 on a validation split.

The cleaned official training subset is divided into 80% fit and 20% validation data.
Stratification preserves category proportions approximately. A fixed random seed
makes that split reproducible. After selecting a model, we retrain that model on all
the cleaned official training data and evaluate on the separate official test subset.

Exercise: explain why choosing parameters after looking at test errors would make
that test set less trustworthy as an independent performance estimate.

## Lesson 6: Evaluation and errors

Precision asks: of the examples predicted as a category, how many really belong there?
Recall asks: of the examples truly in a category, how many did we find?
F1 balances precision and recall. Macro-F1 gives each category equal weight.

The majority-class baseline helps establish whether the learned model does something
useful beyond always guessing the most common label.

Exercise: inspect five rows of `reports/errors.csv`. Are messages short, ambiguous,
or about similar transfer categories? Write observations rather than inventing a cause.
Treat these as a final diagnostic review. Further model changes should use validation;
report that you have inspected this test set rather than claiming it is still untouched.

## Lesson 7: Saving a model and making a prediction

`joblib.dump` saves the fitted pipeline and metadata. `joblib.load` restores it.
`predict.py` converts a new message using the saved TF-IDF vocabulary and applies the
saved classifier. The app calls the same prediction function, so both interfaces share
the same behavior. Load only artifacts you trust; pickle-style formats can execute code.

The top three scores are **uncalibrated**. A high score is not proof a prediction is
right, especially for another language or an unrelated request. This model always
chooses from ten classes. It has no trained "unknown" category.

Exercise: try "What is the weather today?" and explain why the resulting banking label
is a limitation of a closed-set classifier.

## Your first small improvement

Choose one: add a helpful example to the interface, improve a category description,
or write a clear error analysis. Explain the change in `MY_NOTES.md`.
Later, you can experiment with model parameters using validation data, add an API,
or learn MLflow. The local JSON history is simple experiment logging, not a production
model registry, deployed monitoring service, or automatic drift detection system.
