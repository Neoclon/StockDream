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
    "bithumb": "https://api.bithumb.com/v1/candles/minutes/1"
}

INTERVAL = "1m"  # Interval for Binance
BINANCE_LIMIT = 1000  # Max number of data points per request (Binance)
UPBIT_LIMIT = 200  # Max number of data points per request (Upbit)
BITTHUMB_LIMIT = 200  # Max number of data points per request (Bithumb)

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

# Bithumb에서 데이터 가져오기
def fetch_data_bithumb(symbol, datetime_obj):
    headers = {"Accept": "application/json"}
    params = {
        "market": symbol,
        "to": datetime_obj.strftime('%Y-%m-%dT%H:%M:%S'),  # 정확한 ISO 8601 형식
        "count": BITTHUMB_LIMIT,
    }
    response = requests.get(EXCHANGES["bithumb"], params=params, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching data from Bithumb: {response.status_code} {response.text}")
        response.raise_for_status()
    return response.json()  

# 심볼을 거래소별 포맷으로 변경
def clean_symbol(symbol):
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}"
    elif symbol.startswith("KRW-") or symbol.startswith("BTC-"):
        return f"{symbol.replace('-', '_')}"
    else:
        return symbol

# 폴더 구조 생성
def create_exchange_directories(base_directory, exchange):
    """
    거래소별 폴더 구조 생성
    :param base_directory: 최상위 저장 경로
    :param exchange: 거래소 이름 (e.g., binance, upbit, bithumb)
    :return: 각 파일 저장 디렉토리의 경로
    """
    exchange_directory = os.path.join(base_directory, exchange.capitalize())
    csv_directory = os.path.join(exchange_directory, 'csv')
    graph_directory = os.path.join(exchange_directory, 'graphs')
    table_directory = os.path.join(exchange_directory, 'tables')
    text_directory = os.path.join(exchange_directory, 'texts')

    # 폴더 생성
    for directory in [csv_directory, graph_directory, table_directory, text_directory]:
        os.makedirs(directory, exist_ok=True)

    return {
        "csv": csv_directory,
        "graphs": graph_directory,
        "tables": table_directory,
        "texts": text_directory,
    }

# csv 파일 생성
def save_to_csv(data, symbol, directories, start_date, end_date, exchange):
    clean_symbol_name = clean_symbol(symbol)
    csv_directory = directories["csv"]

    # 파일 이름에 거래소 이름 추가
    file_path = f"{csv_directory}/{exchange.capitalize()}_{clean_symbol_name}_{start_date}_{end_date}_data.csv"

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

# 첫 번째 자릿수를 추출하는 함수
def extract_first_digit(value):
    if value == 0 or pd.isnull(value):
        return None
    abs_value = abs(value)
    while abs_value < 1:  # 1 이상이 될 때까지 스케일 조정
        abs_value *= 10
    abs_value_str = ''.join(filter(str.isdigit, str(abs_value)))  # 숫자만 추출
    if len(abs_value_str) > 0:
        return int(abs_value_str[0])  # 첫 번째 자릿수 추출
    return None

# 두 번째 자릿수를 추출하는 함수
def extract_second_digit(value):
    if value == 0 or pd.isnull(value):
        return None
    abs_value = abs(value)
    while abs_value < 1:  # 1 이상이 될 때까지 스케일 조정
        abs_value *= 10
    abs_value_str = ''.join(filter(str.isdigit, str(abs_value)))  # 숫자만 추출
    if len(abs_value_str) > 1:
        return int(abs_value_str[1])  # 두 번째 자릿수 추출
    return None

# 거래량 상승률 계산 및 벤포드 분석
def calculate_volume_change_rate_and_analyze(data, digit_type="first"):
    data['volume_change_rate'] = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
    if digit_type == "first":
        data['Digit'] = data['volume_change_rate'].apply(extract_first_digit)
        actual_frequencies, benford_dist = first_digit_analysis(data, 'Digit')
        return {"first": (actual_frequencies, benford_dist)}
    elif digit_type == "second":
        data['Digit'] = data['volume_change_rate'].apply(extract_second_digit)
        actual_frequencies, benford_dist = second_digit_analysis(data, 'Digit')
        return {"second": (actual_frequencies, benford_dist)}
    elif digit_type == "both":
        first_digits = data['volume_change_rate'].apply(extract_first_digit)
        first_frequencies, first_benford = first_digit_analysis(data.assign(Digit=first_digits), 'Digit')
        
        second_digits = data['volume_change_rate'].apply(extract_second_digit)
        second_frequencies, second_benford = second_digit_analysis(data.assign(Digit=second_digits), 'Digit')
        
        return {
            "first": (first_frequencies, first_benford),
            "second": (second_frequencies, second_benford)
        }
    else:
        raise ValueError("Invalid digit type. Choose 'first', 'second', or 'both'.")


# Benford analysis for first digit
def first_digit_analysis(data, column):
    digit_counts = data[column].value_counts().sort_index()
    benford_dist = [np.log10(1 + 1 / d) for d in range(1, 10)]
    benford_dist_aligned = pd.Series(benford_dist, index=range(1, 10))  # First digit starts from 1
    actual_frequencies = digit_counts / digit_counts.sum()
    actual_frequencies_aligned = actual_frequencies.reindex(benford_dist_aligned.index, fill_value=0)
    return actual_frequencies_aligned, benford_dist_aligned

# Benford analysis for second digit
def second_digit_analysis(data, column):
    digit_counts = data[column].value_counts().sort_index()
    benford_dist = [sum(np.log10(1 + 1 / (10 * d1 + d2)) for d1 in range(1, 10)) for d2 in range(0, 10)]
    benford_dist_aligned = pd.Series(benford_dist, index=range(0, 10))  # Second digit starts from 0
    actual_frequencies = digit_counts / digit_counts.sum()
    actual_frequencies_aligned = actual_frequencies.reindex(benford_dist_aligned.index, fill_value=0)
    return actual_frequencies_aligned, benford_dist_aligned

# 그래프 그리기
def plot_benford_graph(actual_frequencies, benford_dist, symbol, start_date, end_date, directories, digit_type, exchange):
    clean_symbol_name = clean_symbol(symbol)
    graph_directory = directories["graphs"]

    plt.figure(figsize=(10, 6))

    # Actual Distribution as histogram (bar plot)
    plt.bar(actual_frequencies.index, actual_frequencies.values, alpha=0.5, label='Actual Distribution')

    # Actual Distribution as line plot (connecting points with a dashed line)
    plt.plot(actual_frequencies.index, actual_frequencies.values, 'b-', marker='o', linestyle='--', color='blue')

    # Benford's Distribution as line plot
    plt.plot(benford_dist.index, benford_dist.values, linestyle='-', marker='o', color='red', label='Benford Distribution')

    # Graph titles and labels
    plt.title(f'{exchange.capitalize()} - {clean_symbol_name} - TVR - Benford\'s Law Analysis ({digit_type} Digit) ({start_date} to {end_date})')
    plt.xlabel(f'{digit_type} Digit')
    plt.ylabel('Frequency (Proportion)')
    plt.xticks(range(1, 10))
    plt.legend()

    # Save the graph with exchange in filename
    graph_path = f"{graph_directory}/TVR_{exchange.capitalize()}_{clean_symbol_name}_{start_date}_{end_date}_benford_{digit_type.lower()}_graph.png"
    plt.savefig(graph_path)
    plt.close()

# 표 만들기
def plot_benford_table(actual_frequencies, benford_dist, symbol, start_date, end_date, directories, digit_type, exchange):
    clean_symbol_name = clean_symbol(symbol)
    table_directory = directories["tables"]

    # Align actual frequencies and Benford distribution
    actual_frequencies_rounded = actual_frequencies.round(4)
    benford_dist_rounded = benford_dist.round(4)

    # Create table
    benford_table = pd.DataFrame({
        f'{digit_type} Digit': actual_frequencies.index,
        'Actual Frequency': actual_frequencies_rounded.values,
        'Benford Frequency': benford_dist_rounded.values,
    })

    # Save table as an image with exchange in filename
    table_path = f"{table_directory}/TVR_{exchange.capitalize()}_{clean_symbol_name}_{start_date}_{end_date}_benford_{digit_type.lower()}_table.png"
    plt.figure(figsize=(8, 4))
    table_ax = plt.gca()
    table_ax.axis('off')
    table_plot = table(table_ax, benford_table, loc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.scale(1.2, 1.2)
    plt.title(f'{exchange.capitalize()} - {clean_symbol_name} - TVR - Benford\'s Law Table ({digit_type} Digit) ({start_date} to {end_date})')
    plt.savefig(table_path, bbox_inches='tight')
    plt.close()

# MAC 검정 실행
def perform_mad_test(actual_frequencies, benford_dist, symbol, start_date, end_date, directories, digit_type, exchange):
    clean_symbol_name = clean_symbol(symbol)
    text_directory = directories["texts"]

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

    mad_results_path = f"{text_directory}/TVR_{exchange.capitalize()}_{clean_symbol_name}_{start_date}_{end_date}_mad_{digit_type.lower()}_results.txt"
    with open(mad_results_path, "w") as file:
        file.write(f"TVR MAD Test Results ({start_date} to {end_date}):\n")
        file.write(f"Exchange: {exchange.capitalize()}\n")
        file.write(f"Symbol: {clean_symbol_name}\n")
        file.write(f"Digit Type: {digit_type}\n")
        file.write(f"MAD Value: {mad:.6f}\n")
        file.write(f"Conformity: {conformity}\n")

    print(f"MAD test results saved to: {mad_results_path}")

# 메인 실행 함후
def main():
    # 거래소 선택
    exchange = input("Select the exchange (binance/upbit/bithumb): ").strip().lower()
    if exchange not in EXCHANGES:
        print("Unsupported exchange. Please select 'binance', 'upbit', or 'bithumb'.")
        return

    symbols_input = input("Enter the cryptocurrency symbols (e.g., BTCUSDT, KRW-BTC): ").strip().upper()
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

    digit_choice = input("Do you want to analyze the first, second, or both digits? (first/second/both): ").strip().lower()
    if digit_choice not in ["first", "second", "both"]:
        print("Invalid choice. Please select 'first', 'second', or 'both'.")
        return

    save_directory = './crypto_data'
    directories = create_exchange_directories(save_directory, exchange)

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
                    file_path = save_to_csv(data, symbol, directories, start_datetime, end_datetime, exchange)
                    print(f"Saved data for {symbol} from {exchange}.")
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
                    file_path = save_to_csv(data, symbol, directories, start_datetime, end_datetime, exchange)
                    print(f"Saved data for {symbol} from {exchange}.")
                    
                    last_timestamp = datetime.strptime(data[-1]['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S')
                    current_time = last_timestamp - timedelta(minutes=1)
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching data for {symbol}: {e}. Skipping to next cryptocurrency.")
                    break

        elif exchange == "bithumb":
            current_time = end_dt
            while current_time > start_dt:
                try:
                    data = fetch_data_bithumb(symbol, current_time)
                    if not data:
                        print(f"No data available for {symbol}.")
                        break
                    file_path = save_to_csv(data, symbol, directories, start_datetime, end_datetime, exchange)
                    print(f"Saved data for {symbol} from {exchange}.")
                    
                    last_timestamp = datetime.strptime(data[-1]['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S')
                    current_time = last_timestamp - timedelta(minutes=1)
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching data for {symbol}: {e}. Skipping to next cryptocurrency.")
                    break

        if file_path:
            print(f"\nAnalyzing Benford's Law for {symbol} ({digit_choice.capitalize()} Digit)...")
            try:
                df = pd.read_csv(file_path, header=None, names=["timestamp", "open", "high", "low", "close", "volume", "amount"])
                
                # 거래량 상승률 계산 및 벤포드 분석 수행
                analysis_results = calculate_volume_change_rate_and_analyze(df, digit_choice)

                # 결과 처리
                if "first" in analysis_results:
                    first_actual, first_benford = analysis_results["first"]
                    plot_benford_graph(first_actual, first_benford, symbol, start_datetime, end_datetime, directories, "First", exchange)
                    plot_benford_table(first_actual, first_benford, symbol, start_datetime, end_datetime, directories, "First", exchange)
                    if perform_all_tests == "yes":
                        perform_mad_test(first_actual, first_benford, symbol, start_datetime, end_datetime, directories, "First", exchange)

                if "second" in analysis_results:
                    second_actual, second_benford = analysis_results["second"]
                    plot_benford_graph(second_actual, second_benford, symbol, start_datetime, end_datetime, directories, "Second", exchange)
                    plot_benford_table(second_actual, second_benford, symbol, start_datetime, end_datetime, directories, "Second", exchange)
                    if perform_all_tests == "yes":
                        perform_mad_test(second_actual, second_benford, symbol, start_datetime, end_datetime, directories, "Second", exchange)


            except Exception as e:
                print(f"Error analyzing Benford's Law for {symbol}: {e}")


if __name__ == "__main__":
    main()