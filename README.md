# 📈 Machine Learning Stock Forecasting

A machine learning project for forecasting stock prices using historical market data.  
This repository includes model training, evaluation, and a web-based interface for predictions.

---

## 🚀 Project Overview

This project uses historical stock price data to train a machine learning model that predicts future stock prices.

The workflow includes:

- Data collection
- Data preprocessing & scaling
- Model training
- Model evaluation (RMSE metric)
- Web app interface for predictions

---

## 📂 Repository Structure

```
├── Model.ipynb # Jupyter notebook for training & evaluation
├── stock_model.keras # Saved trained model
├── app.py # Web app for predictions
├── requirements.txt # Required Python dependencies
└── README.md # Project documentation
```

---

## 📊 Model Performance

The model performance is evaluated using:

- **RMSE (Root Mean Squared Error)**

Example:
```
RMSE = 36.08
```

RMSE represents the average prediction error in the same unit as the stock price.

Lower RMSE indicates better model accuracy.

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Streamlit / Flask (depending on your app)

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/avviiiral/Machine-Learning-Stock-Forecasting.git
cd Machine-Learning-Stock-Forecasting
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### ▶️ Running the Application

If using Streamlit:
```
streamlit run app.py
```
If using Flask:
```
python app.py
```
The application will open in your browser.

---

### 📈 How It Works

- Historical stock data is loaded.

- Data is scaled using MinMaxScaler.

- Sequential data is created for time-series prediction.

- The trained deep learning model predicts future prices.

- Predictions are inverse-scaled and displayed.

---

### 🔮 Future Improvements

- Add multiple model comparison (LSTM, GRU, ARIMA)

- Integrate live stock API (Yahoo Finance)

- Add production deployment (Docker / AWS)

- Improve evaluation metrics (MAE, MAPE, R²)

---

### ⚠️ Disclaimer

This project is for educational purposes only.
It is not financial advice. Always do your own research before investing.

---

## Author

**Aviral Goyal** 
- LinkedIn: https://www.linkedin.com/in/avviiiral/    
- GitHub: https://github.com/avviiiral  

---

## License

This is an open-source project. You are free to use and modify it.
