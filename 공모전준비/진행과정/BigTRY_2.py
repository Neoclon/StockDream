import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

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

def fetch_binance_parallel(symbol, start_time, end_time, chunk_size=timedelta(days=30)):
    """
    단일 심볼의 데이터를 시간 범위로 분할하고 병렬 처리로 가져옴.
    """
    current_start_time = start_time
    tasks = []

    with ThreadPoolExecutor() as executor:
        while current_start_time < end_time:
            chunk_end_time = min(current_start_time + chunk_size, end_time)
            tasks.append(
                executor.submit(
                    fetch_data_binance, 
                    symbol, 
                    int(current_start_time.timestamp() * 1000), 
                    int(chunk_end_time.timestamp() * 1000)
                )
            )
            current_start_time = chunk_end_time + timedelta(milliseconds=1)

        # 결과 수집 및 결합
        dataframes = [task.result() for task in tasks]
        return pd.concat(dataframes, ignore_index=True)

def fetch_data_binance(symbol, start_time, end_time):
   
    all_data = []
    current_start_time = start_time

    while current_start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": BINANCE_LIMIT,
            "startTime": current_start_time,
            "endTime": end_time,
        }
        response = requests.get(EXCHANGES["binance"], params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            print(f"No more data available for {symbol} from {current_start_time} to {end_time}.")
            break

        # 데이터 추가
        all_data.extend(data)

        # 마지막 데이터의 타임스탬프 + 1ms로 시작 시간 업데이트
        last_timestamp = data[-1][0]
        current_start_time = last_timestamp + 1

    # 전체 데이터를 DataFrame으로 변환
    try:
        df = pd.DataFrame(all_data)
        df = df.iloc[:, :6]  # 첫 6열만 선택 (timestamp, open, high, low, close, volume)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        df["trade_amount"] = df["close"] * df["volume"]
        return df
    except Exception as e:
        print(f"Error processing Binance data for {symbol}: {e}")
        raise

def fetch_data_upbit(symbol, start_datetime, end_datetime):
  
    all_data = []
    current_end_time = end_datetime

    while current_end_time > start_datetime:
        headers = {"Accept": "application/json"}
        params = {
            "market": symbol,
            "to": current_end_time.strftime('%Y-%m-%dT%H:%M:%S'),
            "count": UPBIT_LIMIT,
        }
        response = requests.get(EXCHANGES["upbit"], params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data:
            print(f"No more data available for {symbol} up to {current_end_time}.")
            break

        all_data.extend(data)

        # 마지막 데이터의 타임스탬프 - 1초로 갱신
        last_timestamp = datetime.strptime(data[-1]['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S')
        current_end_time = last_timestamp - timedelta(seconds=1)

    try:
        # 전체 데이터를 DataFrame으로 변환
        df = pd.DataFrame(all_data)[[
            "candle_date_time_utc", "opening_price", "high_price", "low_price", "trade_price", "candle_acc_trade_volume"
        ]]
        df = df.rename(columns={
            "candle_date_time_utc": "timestamp",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by="timestamp").drop_duplicates(subset=["timestamp"], keep="first")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        df["trade_amount"] = df["close"] * df["volume"]
        return df
    except Exception as e:
        print(f"Error processing Upbit data for {symbol}: {e}")
        raise

def fetch_data_bithumb(symbol, start_datetime, end_datetime):

    all_data = []
    current_end_time = end_datetime

    while current_end_time > start_datetime:
        headers = {"Accept": "application/json"}
        params = {
            "market": symbol,
            "to": current_end_time.strftime('%Y-%m-%dT%H:%M:%S'),
            "count": BITTHUMB_LIMIT,
        }
        response = requests.get(EXCHANGES["bithumb"], params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data:
            print(f"No more data available for {symbol} up to {current_end_time}.")
            break

        all_data.extend(data)

        # 마지막 데이터의 타임스탬프 - 1초로 갱신
        last_timestamp = datetime.strptime(data[-1]['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S')
        current_end_time = last_timestamp - timedelta(seconds=1)

    try:
        # 전체 데이터를 DataFrame으로 변환
        df = pd.DataFrame(all_data)[[
            "candle_date_time_utc", "opening_price", "high_price", "low_price", "trade_price", "candle_acc_trade_volume"
        ]]
        df = df.rename(columns={
            "candle_date_time_utc": "timestamp",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by="timestamp").drop_duplicates(subset=["timestamp"], keep="first")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        df["trade_amount"] = df["close"] * df["volume"]
        return df
    except Exception as e:
        print(f"Error processing Bithumb data for {symbol}: {e}")
        raise

def extract_digit(value, position):
    if value == 0 or pd.isnull(value):
        return None
    abs_value = abs(value)
    while abs_value < 1:
        abs_value *= 10
    abs_value_str = ''.join(filter(str.isdigit, str(abs_value)))
    return int(abs_value_str[position - 1]) if len(abs_value_str) >= position else None

def calculate_target_column(data, analysis_target):
    """
    분석 대상 컬럼 선택
    :param data: 데이터프레임
    :param analysis_target: 분석 대상 (PCR, VCR, TV, TA)
    :return: 선택된 컬럼 이름
    """
    if analysis_target == "PCR":
        data['price_change_rate'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
        return 'price_change_rate'
    elif analysis_target == "VCR":
        data['volume_change_rate'] = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
        return 'volume_change_rate'
    elif analysis_target == "TV":
        return 'volume'
    elif analysis_target == "TA":
        return 'trade_amount'
    else:
        raise ValueError(f"Invalid analysis_target: {analysis_target}")

def analyze_first_digit(data, target_column):
    """
    첫 번째 자릿수에 대한 벤포드 분석 수행
    :param data: 데이터프레임
    :param target_column: 분석 대상 컬럼
    :return: actual_frequencies, benford_dist
    """
    data['First_Digit'] = data[target_column].apply(lambda x: extract_digit(x, 1))
    digit_counts = data['First_Digit'].value_counts().sort_index()
    benford_dist = [np.log10(1 + 1 / d) for d in range(1, 10)]
    benford_dist_series = pd.Series(benford_dist, index=range(1, 10))
    actual_frequencies = digit_counts / digit_counts.sum()
    actual_frequencies_aligned = actual_frequencies.reindex(benford_dist_series.index, fill_value=0)


    return actual_frequencies_aligned, benford_dist_series

def analyze_second_digit(data, target_column):
    """
    두 번째 자릿수에 대한 벤포드 분석 수행
    :param data: 데이터프레임
    :param target_column: 분석 대상 컬럼
    :return: actual_frequencies, benford_dist
    """
    data['Second_Digit'] = data[target_column].apply(lambda x: extract_digit(x, 2))
    digit_counts = data['Second_Digit'].value_counts().sort_index()
    benford_dist = [sum(np.log10(1 + 1 / (10 * d1 + d2)) for d1 in range(1, 10)) for d2 in range(0, 10)]
    benford_dist_series = pd.Series(benford_dist, index=range(0, 10))
    actual_frequencies = digit_counts / digit_counts.sum()
    actual_frequencies_aligned = actual_frequencies.reindex(benford_dist_series.index, fill_value=0)

    return actual_frequencies_aligned, benford_dist_series

def calculate_benford_analysis(data, analysis_target, digit_type="both"):
    """
    Benford 분석을 수행하여 결과를 반환
    :param data: 분석 대상 데이터프레임
    :param analysis_target: 분석 대상 (PCR, VCR, TV, TA)
    :param digit_type: 분석할 digit 유형 (first, second, both)
    :return: 분석 결과 (actual_frequencies, benford_dist)
    """
    results = {}
    target_column = calculate_target_column(data, analysis_target)

    if digit_type in ["first", "both"]:
        results["first"] = analyze_first_digit(data, target_column)

    if digit_type in ["second", "both"]:
        results["second"] = analyze_second_digit(data, target_column)

    return results

def plot_mac_time_series(mac_values, time_labels, symbol, term_days, exchange, digit_type, analysis_target, start_datetime, end_datetime):
    plt.figure(figsize=(12, 6))

    # First Digit MAC Values
    if "first" in mac_values and mac_values["first"]:
        plt.plot(
            time_labels,
            mac_values["first"],
            marker='o',          # 마커 모양
            markersize=3,        # 점 크기 설정
            linestyle='-',       # 선 스타일
            linewidth=1,         # 선 굵기 설정
            color='blue',
            label='First Digit MAC Values'
        )

    # Second Digit MAC Values (조건 추가)
    if "second" in mac_values and mac_values["second"]:
        plt.plot(
            time_labels,
            mac_values["second"],
            marker='o',
            markersize=3,        # 점 크기 설정
            linestyle='-',       # 선 스타일
            linewidth=1,         # 선 굵기 설정
            color='orange',
            label='Second Digit MAC Values'
        )
        
    # 기준선 설정
    if digit_type == "first":
        # First Digit 기준선
        plt.axhline(y=0.006, color='green', linestyle='--', linewidth=1, label='Close Conformity')
        plt.axhline(y=0.012, color='purple', linestyle='--', linewidth=1, label='Acceptable Conformity')
        plt.axhline(y=0.015, color='red', linestyle='--', linewidth=1, label='Marginal Conformity')
    elif digit_type == "second":
        # Second Digit 기준선
        plt.axhline(y=0.008, color='green', linestyle='--', linewidth=1, label='Close Conformity')
        plt.axhline(y=0.010, color='purple', linestyle='--', linewidth=1, label='Acceptable Conformity')
        plt.axhline(y=0.012, color='red', linestyle='--', linewidth=1, label='Marginal Conformity')
    elif digit_type == "both":
        # Both의 새로운 기준선
        plt.axhline(y=0.012, color='purple', linestyle='--', linewidth=1, label='SD Marginal Conformity')
        plt.axhline(y=0.015, color='red', linestyle='--', linewidth=1, label='FD Marginal Conformity')

    plt.title(f'{exchange.capitalize()} - {symbol} - {analysis_target} - {term_days}-Day Term MAC Time Series ({digit_type})')
    plt.xlabel('Date')
    plt.ylabel('MAD Value')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)

    graph_path = f"./crypto_data/Timeseries_data/graphs/{exchange.capitalize()}_{symbol}_{analysis_target}_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day_mac_timeseries_{digit_type}.png"
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()
    print(f"Saved MAC Time Series graph to {graph_path}")

def perform_time_series_benford_analysis(exchange, symbols, start_datetime, end_datetime, term_days, digit_type, analysis_target):
    start_dt, _ = datetime_to_timestamp(start_datetime)
    end_dt, _ = datetime_to_timestamp(end_datetime)
    term_delta = timedelta(days=term_days)

    combined_data = []

    for symbol in symbols:
        symbol = symbol.strip()
        print(f"\nPerforming Benford Analysis for {symbol} on {exchange.capitalize()} - {analysis_target} from {start_datetime} to {end_datetime} in {term_days}-day sliding terms.")

        mac_values = {"first": [], "second": []}
        time_labels = []

        try:
            if exchange == "binance":
                # 병렬 처리 적용된 데이터 수집
                df = fetch_binance_parallel(symbol, start_dt, end_dt)
            else:
                raise NotImplementedError("Parallel processing is only implemented for Binance.")

            if df.empty:
                print(f"No data available for {symbol} from {start_dt} to {end_dt}.")
                continue

            # 벤포드 분석 수행
            analysis_results = calculate_benford_analysis(df, analysis_target, digit_type)

            # 분석 결과 처리
            for digit in ["first", "second"]:
                if digit in analysis_results:
                    actual_frequencies, benford_dist = analysis_results[digit]
                    observed = actual_frequencies.values
                    expected = benford_dist.values
                    mad = np.mean(np.abs(expected - observed))
                    mac_values[digit].append(mad)

                    combined_data.append({
                        "symbol": symbol,
                        "start_date": start_datetime,
                        "end_date": end_datetime,
                        "digit_type": digit,
                        "mad": mad,
                        "target": analysis_target
                    })

            time_labels.append(start_dt)

            # MAC 시계열 그래프 생성
            if mac_values["first"] or mac_values["second"]:
                plot_mac_time_series(mac_values, time_labels, symbol, term_days, exchange, digit_type, analysis_target, start_datetime, end_datetime)

        except Exception as e:
            print(f"Error processing data for {symbol} from {start_datetime} to {end_datetime}: {e}")

    # 결과 저장
    if combined_data:
        combined_df = pd.DataFrame(combined_data)
        combined_csv_path = f"./crypto_data/Timeseries_data/MAC_result/{exchange.capitalize()}_{symbols[0]}_{analysis_target}_MAC_Results_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}.csv"
        os.makedirs(os.path.dirname(combined_csv_path), exist_ok=True)
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Saved combined results to {combined_csv_path}")

def main():
    exchange = input("Select the exchange (binance/upbit/bithumb): ").strip().lower()
    if exchange not in EXCHANGES:
        print("Unsupported exchange.")
        exit()

    symbols_input = input("Enter the cryptocurrency symbols (e.g., BTCUSDT, KRW-BTC): ").strip().upper()
    symbols = symbols_input.split(",")

    start_datetime = input("Enter the overall start date and time (YYYY-MM-DD-HH:MM): ").strip()
    end_datetime = input("Enter the overall end date and time (YYYY-MM-DD-HH:MM): ").strip()

    term_days = int(input("Enter the term length in days (e.g., 14): ").strip())
    digit_type = input("Do you want to analyze the first, second, or both digits? (first/second/both): ").strip().lower()
    analysis_target = input("Enter the analysis target (TA/TV/VCR/PCR): ").strip().upper()

    perform_time_series_benford_analysis(exchange, symbols, start_datetime, end_datetime, term_days, digit_type, analysis_target)


if __name__ == "__main__":
    main()