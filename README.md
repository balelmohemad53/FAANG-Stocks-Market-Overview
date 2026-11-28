# FAANG Stocks Market Analysis & Machine Learning

End-to-end analysis of the FAANG stocks (Meta/Facebook, Amazon, Apple, Netflix, Google): exploratory data analysis, a Power BI report, and an interactive Streamlit app for price and direction prediction.

## Project Overview
- **EDA** of price trends, moving averages (SMA 50/200), daily returns and return correlation between the companies.
- **Power BI report** with the main KPIs: Total Volume, Average Close, Average Range, Average Change.
- **Machine learning** (Streamlit app):
  - Regression: predict the next day's closing price (Linear Regression, Random Forest, XGBoost).
  - Classification: predict whether the price goes up or down the next day (Logistic Regression, Random Forest, XGBoost).
  - XGBoost hyperparameter tuning with GridSearchCV and TimeSeriesSplit.

## Repository Files
| File | Description |
|---|---|
| `FAANG.ipynb` | Main notebook: feature engineering, models and tuning |
| `EDA.ipynb` | Exploratory data analysis |
| `app.py` | Streamlit app (EDA, regression, classification, predictions) |
| `requirements.txt` | Python dependencies |
| `FAANG_Stocks_Market_Overview_Team4.pbix` | Power BI file |
| `FAANG_Stocks_Executive_Presentation.pptx` | Executive presentation |

## Features Used by the Models
`Close_Lag1`, `Close_Lag5`, `SMA_10`, `SMA_50`, `High_Low_Range`, `Daily_Return`

## How to Run the App
```bash
git clone https://github.com/balelmohemad53/FAANG-Stocks-Market-Overview.git
cd FAANG-Stocks-Market-Overview
pip install -r requirements.txt
streamlit run app.py
```
Then upload the cleaned Excel file from the sidebar. Required columns: `Source.Name`, `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit, Power BI

## Notes
Built as a university team project for learning purposes. Predictions are not financial advice.
