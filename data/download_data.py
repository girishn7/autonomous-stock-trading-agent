# data/download_data.py
import yfinance as yf

def download_data(ticker="AAPL", start="2020-01-01", end="2023-01-01"):
    data = yf.download(ticker, start=start, end=end)
    data.to_csv("data/stock_data.csv")
    print("Data downloaded")

if __name__ == "__main__":
    download_data()
