# Nykaa Demand Forecasting

Predicts daily product demand per city and suggests how much stock to reorder.

**Live app:** <paste your Streamlit link here>

## Problem
Too little stock disappoints customers; too much stock costs money. This project forecasts demand so stock can be planned in advance.

## Data
Synthetic daily sales data modelled on Indian retail (2022-2025): 5 Nykaa product types, 3 cities, festivals/sales (Diwali, Pink Friday, etc.), discounts and units sold.

## Method
1. Exploratory data analysis (festival, weekend, month and discount effects)
2. Features: date parts, festival flag, discount, lag features (sales 7/14/28 days ago), rolling averages
3. Time-based split: train 2022-2024, test 2025
4. Models compared: Naive baseline, Linear Regression, Ridge, Random Forest, XGBoost (metric: MAE, RMSE, R2)
5. Final XGBoost model saved and served through a Streamlit app with a reorder alert

## Results
<add your MAE table and 1-2 chart screenshots here>

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `nykaa_forecasting.ipynb`: full notebook (Google Colab)
- `app.py`: Streamlit web app
- `nykaa_xgb.json`, `nykaa_meta.json`: saved model and settings
- `Nykaa_Demand.csv`: dataset
