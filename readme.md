# Wi-Fi Experience Dashboard

A Streamlit dashboard for exploring customer Wi-Fi satisfaction, NPS relationships, monthly trends, segment performance, and qualitative Wi-Fi comments.

## Requirements

- Python 3.10 or later
- CSV survey and qualitative-comment data

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Dashboard locally

From the project root, start the app with:

```bash
streamlit run wifi_dashboard_app.py
```

Streamlit prints the local URL in the terminal, usually `http://localhost:8501`.

## Default Data Files

The app loads these local files by default:

- Monthly survey score files: `data/nps_wifi_score_2026-*.csv`
- Pre-filtered Wi-Fi comment data: `data/nps_wifi_comment_only.csv`

The monthly score files must provide the Wi-Fi score and customer-segment columns used by the dashboard. The qualitative file is expected to contain Wi-Fi comments and may include fields such as `SEG_DEP_DT`, `NET_PROMO_CATG`, and `LIKELIHOOD_RECOMMEND_ORIG_LANG_TXT`.

## Uploading Data

Use the sidebar to temporarily override the default survey and qualitative CSV files. Uploaded data is used only for the current Streamlit session and does not replace local files.

## Repository Notes

Local data, exploratory notebooks, and generated Python artifacts are excluded through `.gitignore`. Files removed from Git tracking with `git rm --cached` remain available on the local machine.
