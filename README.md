
# 🌾 Agricultural Price Prediction using XGBoost

This project builds a regression model to predict the **Modal Price** of agricultural commodities using machine learning. We use XGBoost along with feature engineering on time-series and categorical data to forecast price trends. The model is trained on a dataset of commodity prices from various states, districts, and markets in India.

---

## 📌 Project Highlights

- 🔍 **Target**: Modal Price of commodities  
- 📅 **Time Features**: Year, Month, Day, Day of Week  
- 📍 **Location Features**: State, District, Market  
- 🧪 **ML Model**: XGBoost Regressor  
- 📊 **Metric**: RMSE (Root Mean Squared Error)  
- 💾 **Model Output**: Saved in `.json` format for deployment or inference

---

## 🗂️ Folder Structure

```
project-root/
│
├── backend/
│   └── prediction-model/
│       ├── Dataset/
│       │   └── Agriculture_price_dataset.csv
│       ├── Model/
│       │   └── xgb_price_model.json
│       └── Time Series Prediction Model.ipynb
│
├── README.md
```

---

## 🚀 Getting Started

### 🧰 Prerequisites

Install the following Python libraries:

```bash
pip install pandas xgboost scikit-learn numpy
```

---

### ▶️ Run Training

If you’re using the `.ipynb` notebook:

1. Open the notebook: `Time Series Prediction Model.ipynb`
2. Run all cells to preprocess, train the model, and save it.

---

## 💡 Features Used

- **Datetime Components**:  
  `year`, `month`, `day`, `dayofweek`

- **Categorical Features** (One-hot encoded):  
  `STATE`, `District Name`, `Market Name`, `Commodity`, `Variety`, `Grade`

---

## 📈 Model Performance

Model is evaluated using **Root Mean Squared Error (RMSE)** on a test split (20%).  
The model is saved to:  
```bash
./backend/prediction-model/Model/xgb_price_model.json
```

---

## 📦 How to Use the Saved Model

```python
import xgboost as xgb

# Load saved model
model = xgb.Booster()
model.load_model("xgb_price_model.json")

# Use XGBoost DMatrix for prediction
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)
```

---

## 🌱 Future Improvements

- Hyperparameter tuning with `GridSearchCV`
- Model explainability with SHAP
- Adding external data: weather, rainfall, economic indicators
- Real-time prediction API or dashboard

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---