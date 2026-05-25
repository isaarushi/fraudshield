# fraudshield
An app that lets anyone upload a CSV of transactions and instantly get fraud risk scores powered by machine learning.

💻 Run Locally
bash# 1. Install dependencies
pip install -r requirements.txt

# 1. Run the app
streamlit run app.py
Then open http://localhost:8501 in your browser.

📋 CSV Format
Your CSV must have these columns:
ColumnDescriptionExampleamountTransaction amount350.00hour_of_dayHour (0–23)14day_of_weekDay (0=Mon, 6=Sun)2merchant_categoryCategorygrocerycard_typecredit or debitdebitdistance_from_home_kmDistance in km5previous_txn_amountLast transaction amount300failed_attempts_todayFailed auth attempts0is_international0 or 10account_age_daysAccount age in days1200txn_count_last_24hTransactions in 24h3

The app has a Download Sample CSV button to get a ready-made template.


🧠 How It Works

A Random Forest model is trained on 40,000 synthetic transactions at startup (cached — only trains once)
Your uploaded CSV is feature-engineered (time flags, velocity flags, risk score etc.)
Each transaction gets a fraud probability (0–100%), risk level, and recommended action
Results are shown in charts + a filterable table, downloadable as CSV


Risk Levels
LevelProbabilityAction🟢 LOW< 30%APPROVE🟡 MEDIUM30–60%MONITOR🟠 HIGH60–80%REVIEW🔴 CRITICAL> 80%BLOCK
