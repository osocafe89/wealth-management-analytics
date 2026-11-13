# Wealth Management Analytics

An interactive Python application for exploring operational issues and loss-reporting data. Built with Streamlit, the application turns two Excel-based reporting datasets into historical summaries, category trend projections, keyword analysis, and downloadable visualizations.

> **Portfolio project:** This repository presents an independent analytics demonstration. It is not an official product of, or affiliated with, any financial institution. Trend projections and status labels are exploratory reporting aids, not validated risk predictions or financial advice.

## Features

- **Two-report analysis:** Upload an Issues report and a Losses report (`.xlsx`).
- **Shared filters:** Explore selected reporting dates and available regions/locations across the reports.
- **Historical summaries:** Review report counts, monthly averages, peak periods, common categories, and concentrations across advisors or representatives.
- **Keyword analysis:** Extract frequent terms and phrases from free-text descriptions and display their trends over time.
- **Category projections:** Extend monthly category-volume trends three months forward using linear regression; categorize patterns with simple descriptive thresholds.
- **Advisor and account views:** Compare category distributions across higher-volume representatives and accounts, and inspect individual representative trends.
- **Downloadable charts:** Export supported visualizations as PNG files.

The underlying analysis pipeline also calculates exploratory K-means account groups. The current Streamlit interface does not display the group profiles directly.

## Technology

- Python
- Streamlit
- pandas and NumPy
- scikit-learn (`CountVectorizer`, `LinearRegression`, `KMeans`, `StandardScaler`)
- Matplotlib and Seaborn
- openpyxl for Excel file support

## Run locally

1. Clone the repository and change into its directory.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown by Streamlit.
5. Upload both an Issues report and a Losses report in the sidebar to explore the dashboards.

**Current data requirement:** The uploaded spreadsheets must match the expected column names and formats in `app.py`. The current application requires *both* files before the analysis is shown. Built-in fictional sample data is planned for the public portfolio demonstration but has **not yet been added** to this version.

## Deployment

The application can be deployed with [Streamlit Community Cloud](https://share.streamlit.io/):

1. Push `app.py`, `requirements.txt`, and this `README.md` to GitHub.
2. In Streamlit Community Cloud, choose the repository and set the entry point to `app.py`.
3. Deploy the app and test it with non-confidential sample spreadsheets.
4. Link to the public app from a portfolio page in Framer, or embed it if embedding is supported.

## Data privacy

Do not upload or commit real customer information, account numbers, client records, proprietary reports, or other confidential financial data. Public demonstrations should use clearly fictional or appropriately synthesized data only.

## Limitations

- Forecasts are simple three-month linear-trend extrapolations; they do not account for seasonality, external events, uncertainty intervals, or model validation.
- The colored trend-status labels are heuristic descriptions, not validated risk scores.
- K-means clusters are exploratory and should not be interpreted as verified customer segments.
- Dataset quality, missing values, and report-schema differences may affect analysis or prevent a report from loading.

## Project status

This repository is being prepared for a public portfolio demo. Planned improvements include a one-click fictional sample-data mode, more robust upload validation, and removal of institution-specific references from public-facing assets.

## License

No open-source license has been specified. All rights are reserved unless a license is added to this repository.
