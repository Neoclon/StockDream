import requests
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 거래소별 API 엔드포인트 정의
EXCHANGES = {
    "binance": "https://api.binance.com/api/v3/klines",
    "upbit": "https://api.upbit.com/v1/candles/minutes/1",
    "bithumb": "https://api.bithumb.com/v1/candles/minutes/1"
}

INTERVAL = "1m"
BINANCE_LIMIT = 1000
UPBIT_LIMIT = 200
BITTHUMB_LIMIT = 200

def datetime_to_timestamp(datetime_str):
    dt = datetime.strptime(datetime_str, "%Y-%m-%d-%H:%M")
    return dt, int(dt.timestamp() * 1000)

def fetch_data_binance(symbol, start_time, end_time):
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": BINANCE_LIMIT,
        "startTime": start_time,
        "endTime": end_time,
    }
    response = requests.get(EXCHANGES["binance"], params=params)
    response.raise_for_status()
    return response.json()

def fetch_data_upbit(symbol, datetime_obj):
    headers = {"Accept": "application/json"}
    params = {
        "market": symbol,
        "to": datetime_obj.strftime('%Y-%m-%dT%H:%M:%S'),
        "count": UPBIT_LIMIT,
    }
    response = requests.get(EXCHANGES["upbit"], params=params, headers=headers)
    response.raise_for_status()
    return response.json()

def fetch_data_bithumb(symbol, datetime_obj):
    headers = {"Accept": "application/json"}
    params = {
        "market": symbol,
        "to": datetime_obj.strftime('%Y-%m-%dT%H:%M:%S'),
        "count": BITTHUMB_LIMIT,
    }
    response = requests.get(EXCHANGES["bithumb"], params=params, headers=headers)
    response.raise_for_status()
    return response.json()

def save_to_csv(data, symbol, directories, start_datetime, end_datetime, exchange):
    clean_symbol_name = symbol.replace('/', '_').replace('-', '_')
    csv_directory = directories["csv"]
    os.makedirs(csv_directory, exist_ok=True)
    file_path = f"{csv_directory}/{exchange.capitalize()}_{clean_symbol_name}_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_data.csv"
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for kline in data:
            if isinstance(kline, dict) and 'candle_date_time_utc' in kline:
                writer.writerow([
                    kline["candle_date_time_utc"],
                    kline["opening_price"],
                    kline["high_price"],
                    kline["low_price"],
                    kline["trade_price"],
                    kline["candle_acc_trade_volume"],
                    kline["trade_price"] * kline["candle_acc_trade_volume"]
                ])
            elif isinstance(kline, list) and len(kline) >= 6:
                writer.writerow([
                    datetime.utcfromtimestamp(kline[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    float(kline[1]), float(kline[2]), float(kline[3]),
                    float(kline[4]), float(kline[5]),
                    float(kline[4]) * float(kline[5])
                ])
    return file_path

def extract_digit(value, position):
    if value == 0 or pd.isnull(value):
        return None
    abs_value = abs(value)
    while abs_value < 1:
        abs_value *= 10
    abs_value_str = ''.join(filter(str.isdigit, str(abs_value)))
    return int(abs_value_str[position - 1]) if len(abs_value_str) >= position else None

def calculate_price_change_rate_and_analyze(data, digit_type="both"):
    data['price_change_rate'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    results = {}

    if digit_type in ["first", "both"]:
        data['First_Digit'] = data['price_change_rate'].apply(lambda x: extract_digit(x, 1))
        actual_frequencies = data['First_Digit'].value_counts(normalize=True).sort_index()
        benford_dist = pd.Series([np.log10(1 + 1 / d) for d in range(1, 10)], index=range(1, 10))
        results["first"] = (actual_frequencies, benford_dist)

    if digit_type in ["second", "both"]:
        data['Second_Digit'] = data['price_change_rate'].apply(lambda x: extract_digit(x, 2))
        actual_frequencies = data['Second_Digit'].value_counts(normalize=True).sort_index()
        benford_dist = pd.Series([sum(np.log10(1 + 1 / (10 * d1 + d2)) for d1 in range(1, 10)) for d2 in range(0, 10)], index=range(0, 10))
        results["second"] = (actual_frequencies, benford_dist)

    return results

def plot_mac_time_series(mac_values, time_labels, symbol, term_days, exchange, digit_type, directories, start_datetime, end_datetime):
    plt.figure(figsize=(12, 6))
    plt.plot(time_labels, mac_values, marker='o', linestyle='-', label=f'MAC Values ({digit_type.capitalize()} Digit)')
    plt.axhline(y=0.006, color='g', linestyle='--', label='Close Conformity')
    plt.axhline(y=0.012, color='orange', linestyle='--', label='Acceptable Conformity')
    plt.axhline(y=0.015, color='r', linestyle='--', label='Marginal Conformity')
    plt.title(f'{exchange.capitalize()} - {symbol} - {term_days}-Day Term MAC Time Series ({digit_type.capitalize()} Digit)')
    plt.xlabel('Date')
    plt.ylabel('MAD Value')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    graph_path = f"{directories['graphs']}/{exchange.capitalize()}_{symbol}_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day_term_mac_timeseries_{digit_type.lower()}.png"
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()
    print(f"Saved MAC Time Series graph to {graph_path}")

def perform_time_series_benford_analysis(exchange, symbols, start_datetime, end_datetime, term_days, digit_type, directories):
    start_dt, _ = datetime_to_timestamp(start_datetime)
    end_dt, _ = datetime_to_timestamp(end_datetime)
    term_delta = timedelta(days=term_days)

    combined_data = []

    for symbol in symbols:
        symbol = symbol.strip()
        print(f"\nPerforming Benford Analysis for {symbol} on {exchange.capitalize()} from {start_datetime} to {end_datetime} in {term_days}-day terms.")
        current_start = start_dt
        mac_values = []
        time_labels = []

        while current_start + term_delta <= end_dt:
            current_end = current_start + term_delta
            try:
                if exchange == "binance":
                    data = fetch_data_binance(symbol, int(current_start.timestamp() * 1000), int(current_end.timestamp() * 1000))
                elif exchange == "upbit":
                    data = fetch_data_upbit(symbol, current_end)
                elif exchange == "bithumb":
                    data = fetch_data_bithumb(symbol, current_end)

                if not data:
                    print(f"No data available for {symbol} from {current_start} to {current_end}.")
                    current_start += timedelta(days=1)
                    continue

                file_path = save_to_csv(data, symbol, directories, current_start.strftime('%Y-%m-%d-%H:%M'), current_end.strftime('%Y-%m-%d-%H:%M'), exchange)
                df = pd.read_csv(file_path, header=None, names=["timestamp", "open", "high", "low", "close", "volume", "amount"])
                analysis_results = calculate_price_change_rate_and_analyze(df, digit_type)

                for digit, (actual_frequencies, benford_dist) in analysis_results.items():
                    observed = actual_frequencies.values
                    expected = benford_dist.values
                    mad = np.mean(np.abs(expected - observed))
                    mac_values.append(mad)
                    time_labels.append(current_start)
                    combined_data.append({
                        "symbol": symbol,
                        "start_date": current_start.strftime('%Y-%m-%d'),
                        "end_date": current_end.strftime('%Y-%m-%d'),
                        "digit_type": digit,
                        "mad": mad
                    })

            except Exception as e:
                print(f"Error processing data for {symbol} from {current_start} to {current_end}: {e}")

            current_start += timedelta(days=1)

        if mac_values:
            plot_mac_time_series(mac_values, time_labels, symbol, term_days, exchange, digit_type, directories, start_datetime, end_datetime)

    if combined_data:
        combined_df = pd.DataFrame(combined_data)
        combined_csv_path = f"{directories['MAC_Result']}/{exchange.capitalize()}_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day_{digit_type.lower()}_MAC_Results.csv"
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Saved combined results to {combined_csv_path}")

def main():
    exchange = input("Select the exchange (binance/upbit/bithumb): ").strip().lower()
    if exchange not in EXCHANGES:
        print("Unsupported exchange.")
        return

    symbols_input = input("Enter the cryptocurrency symbols (e.g., BTCUSDT, KRW-BTC): ").strip().upper()
    symbols = symbols_input.split(",")

    start_datetime = input("Enter the overall start date and time (YYYY-MM-DD-HH:MM): ").strip()
    end_datetime = input("Enter the overall end date and time (YYYY-MM-DD-HH:MM): ").strip()

    term_days = int(input("Enter the term length in days (e.g., 14): ").strip())
    digit_type = input("Do you want to analyze the first, second, or both digits? (first/second/both): ").strip().lower()

    save_directory_graph = './crypto_data/Timeseries_data/graphs'
    save_directory_csv = './crypto_data/Timeseries_data/csv'
    save_directory_MACResult = './crypto_data/Timeseries_data/MAC_Result'
    directories = {"graphs": save_directory_graph, "csv": save_directory_csv, "MAC_Result": save_directory_MACResult }
    os.makedirs(save_directory_graph, exist_ok=True)

    perform_time_series_benford_analysis(exchange, symbols, start_datetime, end_datetime, term_days, digit_type, directories)

if __name__ == "__main__":
    main()
