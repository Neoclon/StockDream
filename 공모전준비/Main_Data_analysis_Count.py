import requests
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
from scipy.stats import chisquare
from datetime import datetime, timedelta

# 거래소별 API 엔드포인트 정의
EXCHANGES = {
    "binance": "https://api.binance.com/api/v3/klines",
    "upbit": "https://api.upbit.com/v1/candles/minutes/1",
}

INTERVAL = "1m"  # Interval for Binance
BINANCE_LIMIT = 1000  # Max number of data points per request (Binance)
UPBIT_LIMIT = 200  # Max number of data points per request (Upbit)

# 날짜와 시간 문자열을 타임스탬프로 변환하는 함수
def datetime_to_timestamp(datetime_str):
    dt = datetime.strptime(datetime_str, "%Y-%m-%d-%H:%M")
    return dt, int(dt.timestamp() * 1000)

# Binance에서 데이터 가져오기
def fetch_data_binance(symbol, start_time, end_time):
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": BINANCE_LIMIT,
        "startTime": start_time,  # 타임스탬프 사용
        "endTime": end_time,  # 타임스탬프 사용
    }
    response = requests.get(EXCHANGES["binance"], params=params)
    response.raise_for_status()
    return response.json()

# Upbit에서 데이터 가져오기
def fetch_data_upbit(symbol, datetime_obj):
    headers = {"Accept": "application/json"}
    params = {
        "market": symbol,
        "to": datetime_obj.strftime('%Y-%m-%dT%H:%M:%S'),  # 정확한 ISO 8601 형식
        "count": UPBIT_LIMIT,
    }
    response = requests.get(EXCHANGES["upbit"], params=params, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching data from Upbit: {response.status_code} {response.text}")
        response.raise_for_status()
    return response.json()

# 심볼을 거래소별 포맷으로 변경
def clean_symbol(symbol):
    if symbol.endswith("USDT"):
        return f"Binance_{symbol[:-4]}"
    elif symbol.startswith("KRW-"):
        return f"Upbit_{symbol.replace('-', '_')}"
    else:
        return symbol

# CSV 파일에 데이터 저장
def save_to_csv(data, symbol, save_directory, start_date, end_date):
    clean_symbol_name = clean_symbol(symbol)
    csv_directory = os.path.join(save_directory, 'csv')
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    file_path = f"{csv_directory}/{clean_symbol_name}_{start_date}_{end_date}_data.csv"
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # 바이낸스와 Upbit 데이터 구조를 처리
        for kline in data:
            if isinstance(kline, dict):  # Upbit 데이터 구조
                writer.writerow([
                    kline["candle_date_time_utc"],
                    kline["opening_price"],
                    kline["high_price"],
                    kline["low_price"],
                    kline["trade_price"],
                    kline["candle_acc_trade_volume"],
                    kline["trade_price"] * kline["candle_acc_trade_volume"]  # 거래대금 계산
                ])
            else:  # 바이낸스 데이터 구조
                writer.writerow([
                    datetime.utcfromtimestamp(kline[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    float(kline[1]), float(kline[2]), float(kline[3]),
                    float(kline[4]), float(kline[5]),
                    float(kline[4]) * float(kline[5])  # 거래대금 계산
                ])
    return file_path

# Benford analysis
def first_digit_analysis(data, column):
    data['First_Digit'] = data[column].apply(lambda x: int(str(x)[0]) if pd.notnull(x) and str(x)[0] != '0' else None)
    digit_counts = data['First_Digit'].value_counts().sort_index()
    benford_dist = [np.log10(1 + 1 / d) for d in range(1, 10)]
    actual_frequencies = digit_counts / digit_counts.sum()
    return actual_frequencies, benford_dist

# Plot and save graph for Benford's Law analysis
def plot_benford_graph(actual_frequencies, benford_dist, symbol, start_date, end_date, save_directory):
    clean_symbol_name = clean_symbol(symbol)
    graph_directory = os.path.join(save_directory, 'graphs')
    if not os.path.exists(graph_directory):
        os.makedirs(graph_directory)
    
    plt.figure(figsize=(10, 6))
    
    # Actual Distribution as histogram (bar plot)
    plt.bar(actual_frequencies.index, actual_frequencies.values, alpha=0.5, label='Actual Distribution')
    
    # Actual Distribution as line plot (connecting points with a dashed line)
    plt.plot(actual_frequencies.index, actual_frequencies.values, 'b-', marker='o', linestyle='--', color='blue')
    
    # Benford's Distribution as line plot
    plt.plot(range(1, 10), benford_dist, 'ro-', label='Benford Distribution', markerfacecolor='red')
    
    # Graph titles and labels
    plt.title(f'{clean_symbol_name} Trade Amount - Benford\'s Law Analysis ({start_date} to {end_date})')
    plt.xlabel('First Digit')
    plt.ylabel('Frequency (Proportion)')
    plt.xticks(range(1, 10))
    plt.legend()
    
    # Save the graph
    graph_path = f"{graph_directory}/{clean_symbol_name}_{start_date}_{end_date}_benford_graph.png"
    plt.savefig(graph_path)
    plt.close()

# Plot and save table for Benford's Law analysis
def plot_benford_table(actual_frequencies, benford_dist, symbol, start_date, end_date, save_directory):
    clean_symbol_name = clean_symbol(symbol)
    table_directory = os.path.join(save_directory, 'tables')
    if not os.path.exists(table_directory):
        os.makedirs(table_directory)
    actual_frequencies_1 = actual_frequencies.round(4)
    benford_dist_1 = [round(f, 4) for f in benford_dist]
    benford_table = pd.DataFrame({
        'First Digit': range(1, 10),
        'Actual Frequency': [actual_frequencies_1.get(i, 0) for i in range(1, 10)],
        'Benford Frequency': benford_dist_1,
    })
    table_path = f"{table_directory}/{clean_symbol_name}_{start_date}_{end_date}_benford_table.png"
    plt.figure(figsize=(8, 4))
    table_ax = plt.gca()
    table_ax.axis('off')
    table_plot = table(table_ax, benford_table, loc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.scale(1.2, 1.2)
    plt.title(f'{clean_symbol_name} Trade Amount - Benford\'s Law Analysis ({start_date} to {end_date})')
    plt.savefig(table_path, bbox_inches='tight')
    plt.close()

# Perform and save Chi-Square test results
def perform_chi_square_test(actual_frequencies, benford_dist, symbol, start_date, end_date, save_directory):
    clean_symbol_name = clean_symbol(symbol)
    observed = [actual_frequencies.get(i, 0) for i in range(1, 10)]
    expected = benford_dist
    chi2_stat, p_val = chisquare(observed, expected)

    text_directory = os.path.join(save_directory, 'texts')
    if not os.path.exists(text_directory):
        os.makedirs(text_directory)
    chi_square_results_path = f"{text_directory}/{clean_symbol_name}_{start_date}_{end_date}_chi_square_results.txt"
    with open(chi_square_results_path, "w") as file:
        file.write(f"Chi-Square Test Results ({start_date} to {end_date}):\n")
        file.write(f"Symbol: {clean_symbol_name}\n")
        file.write(f"Chi-Square Statistic: {chi2_stat}\n")
        file.write(f"P-value: {p_val}\n")
        if p_val < 0.01:
            file.write(f"Conclusion: {clean_symbol_name} trade amount does not follow Benford's Law.\n")
        else:
            file.write(f"Conclusion: {clean_symbol_name} trade amount follows Benford's Law.\n")
    print(f"Chi-Square test results saved to: {chi_square_results_path}")

# Main function
def main():
    # 거래소 선택
    exchange = input("Select the exchange (binance/upbit): ").strip().lower()
    if exchange not in EXCHANGES:
        print("Unsupported exchange. Please select either 'binance' or 'upbit'.")
        return

    symbols_input = input("Enter the cryptocurrency symbols (e.g., BTCUSDT, KRW-BTC): ").strip().upper()
    symbols = symbols_input.split(",")

    if len(symbols) > 50:
        print("You can only enter up to 50 cryptocurrency symbols.")
        return

    start_datetime = input("Enter the start date and time (YYYY-MM-DD-HH:MM): ").strip()
    end_datetime = input("Enter the end date and time (YYYY-MM-DD-HH:MM): ").strip()

    try:
        start_dt, start_ts = datetime_to_timestamp(start_datetime)  # datetime 객체와 타임스탬프 반환
        end_dt, end_ts = datetime_to_timestamp(end_datetime)  # datetime 객체와 타임스탬프 반환
    except ValueError:
        print("Invalid date-time format. Please use the format YYYY-MM-DD-HH:MM.")
        return

    save_directory = './crypto_data'

    perform_all_tests = input("Do you want to perform the Chi-Square test for all cryptocurrencies? (yes or no): ").strip().lower()

    for symbol in symbols:
        symbol = symbol.strip()
        print(f"\nFetching data for {symbol} from {start_datetime} to {end_datetime}...")
        file_path = None

        if exchange == "binance":
            current_time = start_ts
            while current_time < end_ts:
                try:
                    data = fetch_data_binance(symbol, current_time, min(current_time + BINANCE_LIMIT * 60000, end_ts))
                    if not data:
                        print(f"No data available for {symbol}.")
                        break
                    file_path = save_to_csv(data, symbol, save_directory, start_datetime, end_datetime)
                    print(f"Data for {symbol} saved to: {file_path}")
                    current_time += BINANCE_LIMIT * 60000
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {e}")
                    break
        # 업비트 데이터 저장 루프
        elif exchange == "upbit":
            current_time = end_dt
            while current_time > start_dt:
                try:
                    data = fetch_data_upbit(symbol, current_time)
                    if not data:
                        print(f"No data available for {symbol}.")
                        break
                    file_path = save_to_csv(data, symbol, save_directory, start_datetime, end_datetime)
                    print(f"Saved data for {symbol}.")
                    
                    # 다음 요청의 `to` 시간 업데이트
                    last_timestamp = datetime.strptime(data[-1]['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S')
                    current_time = last_timestamp - timedelta(minutes=1)
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching data for {symbol}: {e}. Skipping to next cryptocurrency.")
                    break

        if file_path:
            print(f"\nAnalyzing Benford's Law for {symbol}...")
            try:
                df = pd.read_csv(file_path, header=None, names=["timestamp", "open", "high", "low", "close", "volume", "amount"])
                actual_frequencies, benford_dist = first_digit_analysis(df, "amount")

                plot_benford_graph(actual_frequencies, benford_dist, symbol, start_datetime, end_datetime, save_directory)
                plot_benford_table(actual_frequencies, benford_dist, symbol, start_datetime, end_datetime, save_directory)

                if perform_all_tests == "yes":
                    perform_chi_square_test(actual_frequencies, benford_dist, symbol, start_datetime, end_datetime, save_directory)

            except Exception as e:
                print(f"Error analyzing Benford's Law for {symbol}: {e}")

if __name__ == "__main__":
    main()
