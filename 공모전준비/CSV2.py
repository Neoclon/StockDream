import requests
import csv
import time
from datetime import datetime
import os

# Binance API endpoint for historical klines
BASE_URL = "https://api.binance.com/api/v3/klines"

# 추가할 거래량 많은 코인 심볼들
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", 
    "BNBUSDT", "SOLUSDT", "ADAUSDT", "LTCUSDT", 
    "DOTUSDT", "AVAXUSDT", "SHIBUSDT", 
    "TRXUSDT", "LINKUSDT", "BCHUSDT", "XLMUSDT", 
    "FTMUSDT", "VETUSDT", "EOSUSDT", "GRTUSDT"
]  # 거래량이 많은 암호화폐들
INTERVAL = "1m"  # 1-minute interval
LIMIT = 1000  # Number of data points per request (maximum is 1000 on Binance)

# 2024년 1월 1일부터 2024년 12월 31일까지의 시작 시간과 종료 시간 설정
START_DATE = "2024-11-01"
END_DATE = "2024-12-31"

# 날짜 문자열을 밀리초 타임스탬프로 변환하는 함수
def date_to_timestamp(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    timestamp = int(dt.timestamp() * 1000)  # 밀리초 단위
    return timestamp

# 시작 시간과 종료 시간 설정
start_time = date_to_timestamp(START_DATE)
end_time = date_to_timestamp(END_DATE)

def fetch_data(symbol, start_time, end_time):
    """
    Fetches cryptocurrency data from Binance API.
    :param symbol: Trading pair symbol (e.g., 'BTCUSDT')
    :param start_time: Start time in milliseconds.
    :param end_time: End time in milliseconds.
    :return: List of kline data.
    """
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": LIMIT,
        "startTime": start_time,
        "endTime": end_time
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()  # Raise an error if the request fails
    return response.json()

def save_to_csv(data, symbol, save_directory):
    """
    Saves kline data to a CSV file.
    :param data: List of kline data.
    :param symbol: Trading pair symbol for the CSV file name.
    :param save_directory: Directory where the file should be saved.
    """
    # Ensure the directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # File path for saving data
    file_name = f"{save_directory}/{symbol.replace('USDT', '').lower()}_data_2024_11_12.csv"
    
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        for kline in data:
            writer.writerow([
                datetime.utcfromtimestamp(kline[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),  # Open time
                float(kline[1]),  # Open price
                float(kline[2]),  # High price
                float(kline[3]),  # Low price
                float(kline[4]),  # Close price
                float(kline[5]),  # Volume
            ])

def main():
    print("Starting data collection...")

    save_directory = '/Users/an-youbin/Desktop/StockDream/SCV_File'  # 원하는 경로 지정

    # Data collection loop for each symbol (16개 코인)
    for symbol in SYMBOLS:
        # Write header if file doesn't exist
        with open(f"{save_directory}/{symbol.replace('USDT', '').lower()}_data_2024_11_12.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Open", "High", "Low", "Close", "Volume"])

        current_time = start_time
        while current_time < end_time:
            try:
                data = fetch_data(symbol, current_time, min(current_time + LIMIT * 60000, end_time))  # 60,000 ms per minute
                if not data:
                    print(f"No new data for {symbol}. Waiting...")
                    break

                save_to_csv(data, symbol, save_directory)
                print(f"Saved {len(data)} records for {symbol}")

                # Update current_time for the next request
                current_time = int(data[-1][0]) + 1

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {symbol}: {e}. Retrying in 10 seconds...")
                time.sleep(10)

if __name__ == "__main__":
    main()
