import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import datetime as dt
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import ta
import os

# Configure Streamlit page
st.set_page_config(page_title="Market Insight Dashboard", layout="wide")
st.title("Market Insight Dashboard")

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("Stock Selection")

# Take primary stock input with examples
# Predefined list of popular stocks
stock_list = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    "TSLA",  # Tesla
    "META",  # Meta
    "NVDA",  # Nvidia
    "NFLX",  # Netflix
    "AMD",   # AMD
    "INTC",  # Intel
    "JPM",   # JPMorgan
    "V",     # Visa
    "WMT",   # Walmart
    "DIS",   # Disney
    "BABA"   # Alibaba
]

# Dropdown selector instead of text input
symbol = st.sidebar.selectbox(
    "Select Primary Stock",
    stock_list
)


# Optional comparison stock
compare_symbol = st.sidebar.text_input("Compare With (Optional)")

# Forecast horizon slider
future_days = st.sidebar.slider("Forecast Horizon (Days)", 7, 60, 30)

# Retrain button
retrain = st.sidebar.button("Retrain Model")

model_path = "stock_model.keras"

# Retrain model if button clicked
if retrain:
    os.system("python retrain.py")

# Stop app if model file missing
if not os.path.exists(model_path):
    st.error("Model file not found. Please retrain the model.")
    st.stop()

# Load trained model
model = load_model(model_path)

# Clean symbol input
symbol = symbol.strip().upper()

if not symbol:
    st.error("Please enter a valid stock symbol.")
    st.stop()

# ---------------- DATA DOWNLOAD ---------------- #

start = "2015-01-01"
end = dt.datetime.today()

# Download stock data
try:
    data = yf.download(symbol, start=start, end=end, progress=False)
except Exception:
    st.error("Error fetching market data.")
    st.stop()

if data.empty:
    st.error("No data found for this symbol.")
    st.stop()

# Fix possible multi-index columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Extract close prices as 1D series
close = data["Close"].squeeze()

# ---------------- TECHNICAL INDICATORS ---------------- #

# Compute RSI indicator
data["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

# Compute MACD indicator
macd_indicator = ta.trend.MACD(close)
data["MACD"] = macd_indicator.macd()
data["MACD_SIGNAL"] = macd_indicator.macd_signal()

# ---------------- TRAIN / TEST SPLIT ---------------- #

train_size = int(len(close) * 0.8)

# Convert to 2D for scaler compatibility
train = close.iloc[:train_size].values.reshape(-1, 1)
test = close.iloc[train_size - 100:].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
scaler.fit(train)

test_scaled = scaler.transform(test)

# Create sequences for LSTM
x_test, y_test = [], []

for i in range(100, len(test_scaled)):
    x_test.append(test_scaled[i - 100:i])
    y_test.append(test_scaled[i])

x_test = np.array(x_test)
y_test = np.array(y_test)

# ---------------- MODEL PREDICTIONS ---------------- #

predictions = model.predict(x_test, verbose=0)

# Convert back to original price scale
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test)

# Compute performance metrics
rmse = np.sqrt(mean_squared_error(actual, predictions))
mape = mean_absolute_percentage_error(actual, predictions) * 100
confidence = max(0, 100 - mape)

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAPE", f"{mape:.2f}%")
col3.metric("Model Confidence", f"{confidence:.1f}%")

# ---------------- PERFORMANCE PLOT ---------------- #

st.subheader("Model Performance")

fig1 = plt.figure(figsize=(12, 6))
plt.plot(actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.legend()
st.pyplot(fig1)

# ---------------- FUTURE FORECAST ---------------- #

# Take last 100 days and reshape properly
last_100 = close.tail(100).values.reshape(-1, 1)
last_100_scaled = scaler.transform(last_100)

future_predictions = []
current_input = last_100_scaled.copy()

# Generate rolling predictions
for _ in range(future_days):
    reshaped = current_input.reshape(1, 100, 1)
    next_pred = model.predict(reshaped, verbose=0)
    future_predictions.append(next_pred[0][0])
    current_input = np.vstack((current_input[1:], next_pred))

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

future_dates = pd.date_range(
    start=data.index[-1], periods=future_days + 1
)[1:]

st.subheader("Forward Projection")

fig2 = plt.figure(figsize=(12, 6))
plt.plot(close, label="Historical Price")
plt.plot(future_dates, future_predictions, label="Forecast")
plt.legend()
st.pyplot(fig2)

# ---------------- SIGNAL LOGIC ---------------- #

st.subheader("Signal Overview")

today_price = close.iloc[-1]
next_price = future_predictions[0][0]

rsi = data["RSI"].iloc[-1]
macd = data["MACD"].iloc[-1]
macd_signal = data["MACD_SIGNAL"].iloc[-1]

score = 0

# Model direction signal
if next_price > today_price:
    score += 1
else:
    score -= 1

# RSI signal
if rsi < 30:
    score += 1
elif rsi > 70:
    score -= 1

# MACD signal
if macd > macd_signal:
    score += 1
else:
    score -= 1

# Final recommendation
if score >= 2:
    st.success("Strong Buy")
elif score == 1:
    st.success("Buy")
elif score == 0:
    st.info("Neutral")
elif score == -1:
    st.warning("Sell")
else:
    st.error("Strong Sell")

st.write(f"RSI: {rsi:.2f}")
st.write(f"MACD: {macd:.2f}")

# ---------------- STRATEGY BACKTEST ---------------- #

st.subheader("Strategy Backtest")

returns = close.pct_change().dropna()

aligned_pred = predictions.flatten()
aligned_actual = actual.flatten()

min_len = min(len(aligned_pred), len(returns))

signals = np.where(aligned_pred[:min_len] > aligned_actual[:min_len], 1, -1)

strategy = returns.iloc[:min_len] * signals

cumulative = (1 + strategy).cumprod()

fig3 = plt.figure(figsize=(12, 6))
plt.plot(cumulative, label="Strategy Equity Curve")
plt.legend()
st.pyplot(fig3)

# ---------------- COMPARISON STOCK ---------------- #

if compare_symbol:
    compare_symbol = compare_symbol.strip().upper()

    try:
        compare_data = yf.download(compare_symbol, start=start, end=end, progress=False)

        if isinstance(compare_data.columns, pd.MultiIndex):
            compare_data.columns = compare_data.columns.get_level_values(0)

        if not compare_data.empty:
            st.subheader("Relative Performance")

            fig4 = plt.figure(figsize=(12, 6))
            plt.plot(close / close.iloc[0], label=symbol)
            plt.plot(compare_data["Close"] / compare_data["Close"].iloc[0], label=compare_symbol)
            plt.legend()
            st.pyplot(fig4)

    except:
        st.warning("Could not fetch comparison stock.")

st.caption("For analytical and educational purposes only.")
