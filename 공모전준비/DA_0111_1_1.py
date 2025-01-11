import requests
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
from datetime import datetime, timedelta

# 거래소별 API 엔드포인트 정의
EXCHANGES = {
    "binance": "https://api.binance.com/api/v3/klines",
    "upbit": "https://api.upbit.com/v1/candles/minutes/1",
    "bithumb": "https://api.bithumb.com/public/candlestick"
}

INTERVAL = "1m"  # Interval for Binance
BINANCE_LIMIT = 1000  # Max number of data points per request (Binance)
UPBIT_LIMIT = 200  # Max number of data points per request (Upbit)
BITTHUMB_LIMIT = 200  # Max number of data points per request (Bithumb)
BITTHUMB_INTERVAL = "1H"  # Interval for Bithumb

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

# Bithumb에서 데이터 가져오기 및 필터링
def fetch_data_bithumb(symbol, start_ts, end_ts):
    """
    빗썸 API에서 데이터를 가져오고, 시간 범위를 나누어 여러 번 요청하여 200개 이상의 데이터를 처리합니다.
    """
    url = f"{EXCHANGES['bithumb']}/{symbol}/{BITTHUMB_INTERVAL}"
    aggregated_data = []  # 모든 데이터를 저장할 리스트
    current_start_ts = start_ts

    while current_start_ts < end_ts:
        params = {
            "start": current_start_ts,
            "end": end_ts,
            "count": BITTHUMB_LIMIT
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching data from Bithumb: {response.status_code} {response.text}")
            response.raise_for_status()

        try:
            raw_data = response.json()["data"]
            if not raw_data:
                print(f"No data available for the period starting {datetime.utcfromtimestamp(current_start_ts / 1000)}")
                break
        except KeyError:
            print("Key 'data' not found in the response")
            break

        # 데이터 필터링
        filtered_data = []
        for row in raw_data:
            try:
                timestamp = int(row[0])  # 타임스탬프
                if current_start_ts <= timestamp <= end_ts:
                    filtered_data.append({
                        "timestamp": timestamp,
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": float(row[5]),
                    })
            except (IndexError, ValueError) as e:
                print(f"Error processing row: {e}, row: {row}")
                continue

        if not filtered_data:
            print(f"No more data available for {symbol} in the range {current_start_ts} to {end_ts}.")
            break

        aggregated_data.extend(filtered_data)

        # 가장 최근 타임스탬프를 기준으로 다음 요청 범위 설정
        current_start_ts = max(item["timestamp"] for item in filtered_data) + 1

    return aggregated_data

# 심볼을 거래소별 포맷으로 변경
def clean_symbol(symbol):
    if symbol.endswith("USDT"):
        return f"Binance_{symbol[:-4]}"
    elif symbol.startswith("KRW-"):
        return f"Upbit_{symbol.replace('-', '_')}"
    elif symbol.endswith("BTC") or symbol.endswith("KRW") or symbol.endswith("ETH"):
        return f"Bithumb_{symbol.replace('-', '_')}"
    else:
        return symbol

def save_to_csv(data, symbol, save_directory, start_date, end_date):
    clean_symbol_name = clean_symbol(symbol)
    csv_directory = os.path.join(save_directory, 'csv')
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    file_path = f"{csv_directory}/{clean_symbol_name}_{start_date}_{end_date}_data.csv"

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        for kline in data:
            if isinstance(kline, dict) and 'candle_date_time_utc' in kline:  # Upbit 데이터 구조
                writer.writerow([
                    kline["candle_date_time_utc"],
                    kline["opening_price"],
                    kline["high_price"],
                    kline["low_price"],
                    kline["trade_price"],
                    kline["candle_acc_trade_volume"],
                    kline["trade_price"] * kline["candle_acc_trade_volume"]  # 거래대금 계산
                ])
            elif isinstance(kline, dict) and 'timestamp' in kline:  # 빗썸 데이터 구조
                writer.writerow([
                    datetime.utcfromtimestamp(kline["timestamp"] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    kline["open"], kline["high"], kline["low"],
                    kline["close"], kline["volume"], kline["close"] * kline["volume"] # 거래대금 계산
                ])
            else:  # 바이낸스 데이터 구조
                writer.writerow([
                    datetime.utcfromtimestamp(kline[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    float(kline[1]), float(kline[2]), float(kline[3]),
                    float(kline[4]), float(kline[5]),
                    float(kline[4]) * float(kline[5])  # 거래대금 계산
                ])
    return file_path

# Benford analysis for first digit
def first_digit_analysis(data, column):
    data['First_Digit'] = data[column].apply(lambda x: int(str(x)[0]) if pd.notnull(x) and str(x)[0] != '0' else None)
    digit_counts = data['First_Digit'].value_counts().sort_index()
    benford_dist = [np.log10(1 + 1 / d) for d in range(1, 10)]
    benford_dist_aligned = pd.Series(benford_dist, index=range(1, 10))  # First digit starts from 1
    actual_frequencies = digit_counts / digit_counts.sum()
    actual_frequencies_aligned = actual_frequencies.reindex(benford_dist_aligned.index, fill_value=0)
    return actual_frequencies_aligned, benford_dist_aligned

# Benford analysis for second digit
def second_digit_analysis(data, column):
    data['Second_Digit'] = data[column].apply(lambda x: int(str(x)[1]) if pd.notnull(x) and len(str(x)) > 1 else None)
    digit_counts = data['Second_Digit'].value_counts().sort_index()
    benford_dist = [sum(np.log10(1 + 1 / (10 * d1 + d2)) for d1 in range(1, 10)) for d2 in range(0, 10)]
    benford_dist_aligned = pd.Series(benford_dist, index=range(0, 10))  # Second digit starts from 0
    actual_frequencies = digit_counts / digit_counts.sum()
    actual_frequencies_aligned = actual_frequencies.reindex(benford_dist_aligned.index, fill_value=0)
    return actual_frequencies_aligned, benford_dist_aligned

def plot_benford_graph(actual_frequencies, benford_dist, symbol, start_date, end_date, save_directory, digit_type):
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
    plt.plot(benford_dist.index, benford_dist.values, linestyle='-', marker='o', color='red', label='Benford Distribution')

    # Graph titles and labels
    plt.title(f'{clean_symbol_name} Trade Amount - Benford\'s Law Analysis ({digit_type} Digit) ({start_date} to {end_date})')
    plt.xlabel(f'{digit_type} Digit')
    plt.ylabel('Frequency (Proportion)')
    plt.xticks(range(len(benford_dist)))
    plt.legend()

    # Save the graph
    graph_path = f"{graph_directory}/{clean_symbol_name}_{start_date}_{end_date}_benford_{digit_type.lower()}_graph.png"
    plt.savefig(graph_path)
    plt.close()

def plot_benford_table(actual_frequencies, benford_dist, symbol, start_date, end_date, save_directory, digit_type):
    clean_symbol_name = clean_symbol(symbol)
    table_directory = os.path.join(save_directory, 'tables')
    if not os.path.exists(table_directory):
        os.makedirs(table_directory)

    # Align actual frequencies and Benford distribution
    actual_frequencies_rounded = actual_frequencies.round(4)
    benford_dist_rounded = benford_dist.round(4)

    # Create table
    benford_table = pd.DataFrame({
        f'{digit_type} Digit': actual_frequencies.index,
        'Actual Frequency': actual_frequencies_rounded.values,
        'Benford Frequency': benford_dist_rounded.values,
    })

    # Save table as an image
    table_path = f"{table_directory}/{clean_symbol_name}_{start_date}_{end_date}_benford_{digit_type.lower()}_table.png"
    plt.figure(figsize=(8, 4))
    table_ax = plt.gca()
    table_ax.axis('off')
    table_plot = table(table_ax, benford_table, loc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.scale(1.2, 1.2)
    plt.title(f'{clean_symbol_name} Trade Amount - Benford\'s Law Table ({digit_type} Digit) ({start_date} to {end_date})')
    plt.savefig(table_path, bbox_inches='tight')
    plt.close()

def perform_mad_test(actual_frequencies, benford_dist, symbol, start_date, end_date, save_directory, digit_type):
    clean_symbol_name = clean_symbol(symbol)

    # Ensure actual frequencies and Benford distribution are aligned
    observed = actual_frequencies.values
    expected = benford_dist.values

    # Calculate MAD
    mad = np.mean(np.abs(expected - observed))

    # Determine conformity
    if mad <= 0.006:
        conformity = "Close Conformity"
    elif mad <= 0.012:
        conformity = "Acceptable Conformity"
    elif mad <= 0.015:
        conformity = "Marginally Acceptable Conformity"
    else:
        conformity = "Non-Conformity"

    # Save results to a text file
    text_directory = os.path.join(save_directory, 'texts')
    if not os.path.exists(text_directory):
        os.makedirs(text_directory)

    mad_results_path = f"{text_directory}/{clean_symbol_name}_{start_date}_{end_date}_mad_{digit_type.lower()}_results.txt"
    with open(mad_results_path, "w") as file:
        file.write(f"MAD Test Results ({start_date} to {end_date}):\n")
        file.write(f"Symbol: {clean_symbol_name}\n")
        file.write(f"Digit Type: {digit_type}\n")
        file.write(f"MAD Value: {mad:.6f}\n")
        file.write(f"Conformity: {conformity}\n")

    print(f"MAD test results saved to: {mad_results_path}")

# Main function
def main():
    # 거래소 선택
    exchange = input("Select the exchange (binance/upbit/bithumb): ").strip().lower()
    if exchange not in EXCHANGES:
        print("Unsupported exchange. Please select 'binance', 'upbit', or 'bithumb'.")
        return

    symbols_input = input("Enter the cryptocurrency symbols (e.g., BTCUSDT, KRW-BTC, BTC_KRW): ").strip().upper()
    symbols = symbols_input.split(",")

    if len(symbols) > 50:
        print("You can only enter up to 50 cryptocurrency symbols.")
        return

    start_datetime = input("Enter the start date and time (YYYY-MM-DD-HH:MM): ").strip()
    end_datetime = input("Enter the end date and time (YYYY-MM-DD-HH:MM): ").strip()

    try:
        start_dt, start_ts = datetime_to_timestamp(start_datetime)
        end_dt, end_ts = datetime_to_timestamp(end_datetime)
    except ValueError:
        print("Invalid date-time format. Please use the format YYYY-MM-DD-HH:MM.")
        return

    digit_choice = input("Do you want to analyze the first or second digit? (first/second): ").strip().lower()
    if digit_choice not in ["first", "second"]:
        print("Invalid choice. Please select 'first' or 'second'.")
        return

    save_directory = './crypto_data'

    perform_all_tests = input("Do you want to perform the MAD test for all cryptocurrencies? (yes or no): ").strip().lower()

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
                    
                    last_timestamp = datetime.strptime(data[-1]['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S')
                    current_time = last_timestamp - timedelta(minutes=1)
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching data for {symbol}: {e}. Skipping to next cryptocurrency.")
                    break

        elif exchange == "bithumb":
            try:
                # 빗썸 데이터 가져오기
                data = fetch_data_bithumb(symbol, start_ts, end_ts)
                if not data:
                    print(f"No data available for {symbol}.")
                    continue

                # CSV 저장
                file_path = save_to_csv(data, symbol, save_directory, start_datetime, end_datetime)
                print(f"Data for {symbol} saved to: {file_path}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue

        if file_path:
            print(f"\nAnalyzing Benford's Law for {symbol} ({digit_choice.capitalize()} Digit)...")
            try:
                df = pd.read_csv(file_path, header=None, names=["timestamp", "open", "high", "low", "close", "volume", "amount"])
                if digit_choice == "first":
                    actual_frequencies, benford_dist = first_digit_analysis(df, "amount")
                else:
                    actual_frequencies, benford_dist = second_digit_analysis(df, "amount")

                plot_benford_graph(actual_frequencies, benford_dist, symbol, start_datetime, end_datetime, save_directory, digit_choice.capitalize())
                plot_benford_table(actual_frequencies, benford_dist, symbol, start_datetime, end_datetime, save_directory, digit_choice.capitalize())

                if perform_all_tests == "yes":
                    perform_mad_test(actual_frequencies, benford_dist, symbol, start_datetime, end_datetime, save_directory, digit_choice.capitalize())

            except Exception as e:
                print(f"Error analyzing Benford's Law for {symbol}: {e}")

if __name__ == "__main__":
    main()
