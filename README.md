# Nykaa Demand Forecasting

Predicts daily product demand per city and suggests how much stock to reorder.

**Live app:** https://nykaa-demand-forecasting-cmgw93ppipvtzawqsej2ek.streamlit.app/

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
## Results

Models were trained on 2022-2024 and tested on 2025 (time-based split).

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| XGBoost | 8.53 | 11.33 | 0.934 |
| Random Forest | 8.76 | 11.86 | 0.928 |
| Ridge | 10.94 | 16.06 | 0.868 |
| Linear Regression | 10.94 | 16.06 | 0.868 |
| Naive (last week) | 25.14 | 41.30 | 0.124 |

XGBoost reduced the average error (MAE) by about 66% compared with the
naive "same as last week" baseline, and by about 22% compared with Ridge.
Random Forest came very close to XGBoost, so the gap between the two is small.

Note: the data is synthetic, so these scores show that the pipeline works,
not how the model would perform on real Nykaa sales.

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
