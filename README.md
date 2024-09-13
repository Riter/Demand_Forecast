# Demand Forecast
Demand Forecast service using quantile regression

## Project Organization
------------

├── README.md
├── app.py      <--- FastAPI app
├── data
│   ├── get_data.sql
│   ├── processed   <--- The final, canonical data sets for modeling
│   │   └── predictions.csv
│   └── raw     <--- The original, immutable data dump
│       ├── demand_orders.csv   <--- Raw about orders
│       ├── sales.csv   <--- Raw aggregated data about orders
│       └── sales_with_features.csv
├── evaluate.py     <--- Evaluating model performance
├── features.py     <--- Adding features to sales.csv
├── inference.py    <--- ClearML pipeline for predicting
├── model.py        <--- Class for fitting and prediction
├── models      <--- Trained models
│   └── model.pkl
├── notebooks
│   └── EDA.ipynb
└── training.py    <--- ClearML pipeline for model training

--------