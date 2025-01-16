import yfinance as yf

# Define the ticker for SPDR Gold Shares (GLD)
gold_ticker = "GLD"

# Fetch historical data
try:
    gold_data = yf.download(gold_ticker, start="2020-01-01", end="2025-01-01", interval="1d")

    # Save to CSV
    gold_data.to_csv("gold_data.csv")
    print("Gold data downloaded successfully.")

except Exception as e:
    print(f"Failed to download data: {e}")
