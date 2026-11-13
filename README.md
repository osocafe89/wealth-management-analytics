# Wealth Management Analytics

An independent, neutral Streamlit portfolio demonstration for exploring **fictional** operational Issues and Losses reporting data.

## Launch the interactive demo

The application automatically opens with built-in fictional reports. **No upload is required.** Choose "Upload Excel reports" in the sidebar only if you want to test compatible files.

## Features

- Two-report historical summary and shared date/region filters
- Advisor, account and category volume charts
- Text themes extracted with scikit-learn CountVectorizer
- Illustrative three-month **linear-regression** volume trend extrapolations
- Exploratory K-means clustering computed in the pipeline (not separately exposed in the Streamlit interface)
- Individual advisor scorecards and downloadable PNG charts

This is a portfolio project, not an official banking application. **All bundled records are fictional.** Trend extrapolations are not validated predictions; threshold labels are simple heuristics rather than calibrated risk ratings.

## Files

```text
app.py
requirements.txt
demo_data/
  sample_issues.csv
  sample_losses.csv
.gitignore
README.md
```

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Visit https://share.streamlit.io and sign in with GitHub.
2. Select the `osocafe89/wealth-management-analytics` repository and branch `main`.
3. Set the main file path to `app.py` and deploy.
4. Copy the resulting `*.streamlit.app` URL into an Embed component on your Framer portfolio, and provide an external "Launch demo" link as well.

## Optional custom Excel uploads

Upload both files together. Required columns:

| Issues report | Losses report |
|---|---|
| `Date`, `Advisor`, `Category`, `Issue`, `Account` | `Date`, `Rep Name`, `Type`, `Reason`, `Account` |

The Issues report may also contain `Region` and `Follow up email sent`. Losses may contain `Location`.

**Privacy:** Do not submit or commit genuine client information, proprietary financial records, or regulated information.
