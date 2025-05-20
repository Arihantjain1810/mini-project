from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import os
import base64
from io import BytesIO

app = Flask(__name__,template_folder='templates')

# List of 10 companies
companies = {
    'AAPL': 'Apple Inc.',
    'GOOGL': 'Alphabet Inc.',
    'MSFT': 'Microsoft Corporation',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'TSLA': 'Tesla Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'WMT': 'Walmart Inc.',
    'DIS': 'The Walt Disney Company',
    'NFLX': 'Netflix Inc.'
}

# Create data directory if not exists
if not os.path.exists('stock_data'):
    os.makedirs('stock_data')

def generate_mock_data(ticker, start_date='2020-01-01', end_date='2025-04-30'):
    date_range = pd.date_range(start=start_date, end=end_date)
    days = len(date_range)
    
    base_price = {
        'AAPL': 75, 'GOOGL': 120, 'MSFT': 150, 'AMZN': 90, 
        'META': 180, 'TSLA': 120, 'JPM': 130, 'WMT': 140, 
        'DIS': 100, 'NFLX': 300
    }.get(ticker, 100)
    
    daily_returns = np.random.normal(0.001, 0.02, days)
    price = base_price * (1 + daily_returns).cumprod()
    seasonality = np.sin(np.arange(days) * 2 * np.pi / 252) * 5
    price += seasonality
    price += np.random.normal(0, 2, days)
    price = np.abs(price)
    
    df = pd.DataFrame({
        'Date': date_range,
        'Open': price,
        'High': price * (1 + np.abs(np.random.normal(0.005, 0.01, days))),
        'Low': price * (1 - np.abs(np.random.normal(0.005, 0.01, days))),
        'Close': price * (1 + np.random.normal(0, 0.005, days)),
        'Volume': np.random.randint(1000000, 50000000, days),
        'Company': companies[ticker]
    })
    
    return df

def get_stock_data(ticker):
    csv_path = f'stock_data/{ticker}.csv'
    
    if not os.path.exists(csv_path):
        df = generate_mock_data(ticker)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path, parse_dates=['Date'])
    
    return df

def predict_stock_price(ticker, prediction_date):
    df = get_stock_data(ticker)
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    
    X = df[['Days']]
    y = df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    prediction_date = pd.to_datetime(prediction_date)
    prediction_days = (prediction_date - df['Date'].min()).days
    predicted_price = model.predict([[prediction_days]])[0]
    
    actual_price = None
    if prediction_date in df['Date'].values:
        actual_price = df[df['Date'] == prediction_date]['Close'].values[0]
    
    return actual_price, predicted_price, model, df

def create_plot(ticker, prediction_date, actual, predicted, model, df):
    plt.figure(figsize=(10, 5))
    
    # Plot historical data
    plt.plot(df['Date'], df['Close'], label='Historical Prices', color='blue')
    
    # Highlight 2024-2025 period
    mask = (df['Date'] >= '2024-01-01') & (df['Date'] <= '2025-04-30')
    plt.plot(df.loc[mask, 'Date'], df.loc[mask, 'Close'], color='purple', linewidth=2, label='2024-2025 Prices')
    
    # Plot prediction
    prediction_date_dt = pd.to_datetime(prediction_date)
    plt.scatter(prediction_date_dt, predicted, color='red', s=100, label=f'Predicted: ${predicted:.2f}')
    
    # Plot actual if available
    if actual is not None:
        plt.scatter(prediction_date_dt, actual, color='green', s=100, label=f'Actual: ${actual:.2f}')
    
    # Plot trend line
    future_dates = pd.date_range(start='2024-01-01', end='2025-06-30')
    future_days = (future_dates - df['Date'].min()).days.values.reshape(-1, 1)
    future_prices = model.predict(future_days)
    plt.plot(future_dates, future_prices, '--', color='orange', label='Trend Line')
    
    plt.title(f'{companies[ticker]} ({ticker}) Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['company']
        prediction_date = request.form['prediction_date']
        
        actual, predicted, model, df = predict_stock_price(ticker, prediction_date)
        plot_url = create_plot(ticker, prediction_date, actual, predicted, model, df)
        
        return render_template('index.html', 
                            companies=companies,
                            selected_company=ticker,
                            prediction_date=prediction_date,
                            actual_price=actual,
                            predicted_price=predicted,
                            plot_url=plot_url)
    
    return render_template('index.html', companies=companies)

if __name__ == '__main__':
    app.run(debug=True)