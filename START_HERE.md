# Start here: customer_support_classifier

This project reads an English banking support message and predicts one of ten categories.
The starter implementation, trained model, evaluation reports, and local demo are provided.
Your job is to run it, understand it, and make a small improvement you can explain.

## 1. Open the project

In VS Code, use File > Open Folder and select:

`D:\AI_ML Projects\customer_support_classifier`

Open a PowerShell terminal inside that folder. The `.venv` environment is already set up
on this computer. You can run its Python directly without activating it.

## 2. Try the model

```powershell
& '.\.venv\Scripts\python.exe' predict.py "Someone stole my bank card"
```

Try another message:

```powershell
& '.\.venv\Scripts\python.exe' predict.py "I want to close my account"
```

You will see three candidate categories and their scores. A score is not a verified
probability that the answer is right. The model cannot recognize arbitrary new categories.

## 3. Open the app

Double-click `start_demo.cmd`, or run:

```powershell
& '.\.venv\Scripts\python.exe' -m streamlit run app.py --browser.gatherUsageStats false
```

Open the local URL printed by Streamlit (normally http://localhost:8501).
This is a local demo. It does not establish public deployment.
Keep that terminal running while using the app; press Ctrl+C to stop it.

## 4. Read the results

Open `reports/RESULTS.md`, then `reports/confusion_matrix.png` and `reports/errors.csv`.

- **Accuracy:** what fraction of test messages were classified correctly?
- **Macro-F1:** calculate F1 separately for each category, then average them equally.
- **Confusion matrix:** rows are actual categories; columns are predicted categories.
  Correct predictions lie on the diagonal.
- **Baseline:** a model that always chooses the most frequent training category.

The score measures a selected ten-category English banking task, not general support
classification, all 77 BANKING77 intents, or real-world banking performance.

## 5. Follow the learning guide

Read `LEARN_WITH_ME.md` one lesson at a time. Start with the four-line dictionary example.
You do not need to understand every source file at once.

## 6. Reproduce training when you understand the split

```powershell
& '.\.venv\Scripts\python.exe' train.py
& '.\.venv\Scripts\python.exe' -m pytest tests -q
```

Training regenerates the model and reports and appends a local JSON experiment record.
The raw dataset is reused; its recorded SHA-256 hashes are checked each time.
If you retrain while the app is open, stop and restart the app to clear its model cache.

## Before putting it on your resume

- Run both the command-line predictor and the app yourself.
- Explain TF-IDF, the train/validation/test distinction, and macro-F1 in your own words.
- Inspect five incorrect predictions and record observations in `MY_NOTES.md`.
- Make one small change and describe what you contributed.
- Use the actual score in `reports/RESULTS.md`; describe it as a 10-intent subset.
- Mention this as an AI-assisted learning project when discussing how it was built.
  Describe only the components you understand and have verified yourself.

## If the environment is missing on another computer

Install Python 3.12, then run these from the project folder:

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-dev.txt
& '.\.venv\Scripts\python.exe' train.py
```

The runtime versions were recorded on the machine used to build this project. If a
package is unavailable on your platform, create a separate environment and revalidate
the model and app before changing the recorded dependencies.
