import yfinance as yf

# Define the ticker for SPDR Gold Shares (GLD)
microsoft = "MSF"

# Fetch historical data
try:
    micro_data = yf.download(microsoft, start="2024-01-01", end="2025-01-01", interval="1d")

    # Save to CSV
    micro_data.to_csv("microsoft_data.csv")
    print("Microsoft data downloaded successfully.")

except Exception as e:
    print(f"Failed to download data: {e}")
