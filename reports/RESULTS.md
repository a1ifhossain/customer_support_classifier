# Evaluation results

Run: `20261021T061209Z_5c38e694`  
Selected model: **logistic_regression**, chosen using validation macro-F1.  
Final training messages: **1355**. Held-out test messages: **399**.

| Method | Test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| Majority-class baseline | 0.1003 | 0.0182 |
| TF-IDF + logistic_regression | 0.9549 | 0.9551 |

These results apply only to the selected ten English banking intents after the documented
duplicate audit. They are not a score on the complete 77-intent benchmark or evidence
of performance on live customer requests. See `errors.csv` for every incorrect prediction.

## Resume wording (after you can explain and reproduce the project)

Built a 10-intent banking support classifier using TF-IDF and logistic regression,
achieving 0.955 macro-F1 on 399 held-out messages after duplicate auditing;
compared two models using validation data and built a Streamlit prediction interface.

## Review before claiming completion

Run the app yourself, inspect at least five errors, and describe two limitations in your own words.
Training and demo code do not establish public deployment or production monitoring.
