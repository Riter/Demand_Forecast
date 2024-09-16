# Demand Forecast
Demand Forecast service using quantile regression

## Setup

```
pip install -r requirements.txt
```

## Usage

### Service

Run the application by FastAPI:
```
python src/app.py
```

Run the application by Streamlit:
```
streamlit run web_app/streamlit_app.py
```

Visit http://localhost:5000/docs and upload data/processed/predictions.csv:


For example, you would like to know the demand for the product with sku id 27 for the next 14 days. You should fill in the fields in web service: SKU_ID = 27, Stock = 16, Horizon Days = 14, Confidence Level = 0.90
Сlick the buttons: 
- 3.1 "Get how much to order" to find out how much diawara you need to order from the supplier. 
- 3.2 "Get stock level" to find out how much stock you will have in 14 days. 
- 3.3 "Get low stock sku id" to find out which products will be out of stock in 14 days.


### Pipelines



## Project Organization
------------

```
├── README.md
├── app.py      <--- FastAPI app
├── data
│   ├── get_data.sql
│   ├── processed   <--- The final, canonical data sets for modeling
│   │   └── predictions.csv
│   └── raw     <--- The original, immutable data dump
│       ├── demand_orders.csv   <--- Raw data about orders
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
```

------------

