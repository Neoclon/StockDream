import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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
    data = response.json()

    try:
        # 필요한 열만 선택하여 DataFrame 생성
        df = pd.DataFrame(data)
        df = df.iloc[:, :6]  # 첫 6열만 선택 (timestamp, open, high, low, close, volume)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        print(f"Error processing Binance data for {symbol}: {e}")
        print(f"Returned data: {data[:3]}")  # 첫 3개의 데이터를 출력하여 문제를 파악
        raise

def fetch_data_upbit(symbol, datetime_obj):
    headers = {"Accept": "application/json"}
    params = {
        "market": symbol,
        "to": datetime_obj.strftime('%Y-%m-%dT%H:%M:%S'),
        "count": UPBIT_LIMIT,
    }
    response = requests.get(EXCHANGES["upbit"], params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    try:
        # 필요한 열만 선택하여 DataFrame 생성
        df = pd.DataFrame(data)[[
            "candle_date_time_utc", "opening_price", "high_price", "low_price", "trade_price", "candle_acc_trade_volume"
        ]]

        # 열 이름 변경
        df = df.rename(columns={
            "candle_date_time_utc": "timestamp",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume"
        })

        # 중복 제거 및 정렬
        df = df.drop_duplicates(subset=["timestamp"], keep="first").sort_values(by="timestamp")

        # 데이터 타입 변환
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df

    except Exception as e:
        print(f"Error processing Upbit data for {symbol}: {e}")
        print(f"Returned data: {data[:3]}")  # 반환된 데이터 일부를 출력
        raise

def fetch_data_bithumb(symbol, datetime_obj):
    headers = {"Accept": "application/json"}
    params = {
        "market": symbol,
        "to": datetime_obj.strftime('%Y-%m-%dT%H:%M:%S'),
        "count": BITTHUMB_LIMIT,
    }
    response = requests.get(EXCHANGES["bithumb"], params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    try:
        # 필요한 열만 선택하여 DataFrame 생성
        df = pd.DataFrame(data)[[
            "candle_date_time_utc", "opening_price", "high_price", "low_price", "trade_price", "candle_acc_trade_volume"
        ]]

        # 열 이름 변경
        df = df.rename(columns={
            "candle_date_time_utc": "timestamp",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume"
        })

        # 중복 제거 및 정렬
        df = df.drop_duplicates(subset=["timestamp"], keep="first").sort_values(by="timestamp")

        # 데이터 타입 변환
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df

    except Exception as e:
        print(f"Error processing Bithumb data for {symbol}: {e}")
        print(f"Returned data: {data[:3]}")  # 반환된 데이터 일부를 출력
        raise

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

def plot_mac_time_series(mac_values, time_labels, symbol, term_days, exchange, digit_type, start_datetime, end_datetime):
    """
    First와 Second의 MAC 값을 각각 구분된 색상으로 시각화하며,
    점 크기와 선 굵기를 조정합니다.
    """
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

    plt.title(f'{exchange.capitalize()} - {symbol} - {term_days}-Day Term MAC Time Series ({digit_type})')
    plt.xlabel('Date')
    plt.ylabel('MAD Value')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)

    # 그래프 파일 저장 경로
    graph_path = f"./crypto_data/Timeseries_data/graphs/{exchange.capitalize()}_{symbol}_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day_mac_timeseries_{digit_type}.png"
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()
    print(f"Saved MAC Time Series graph to {graph_path}")

def perform_time_series_benford_analysis(exchange, symbols, start_datetime, end_datetime, term_days, digit_type):
    start_dt, _ = datetime_to_timestamp(start_datetime)
    end_dt, _ = datetime_to_timestamp(end_datetime)
    term_delta = timedelta(days=term_days)

    combined_data = []

    for symbol in symbols:
        symbol = symbol.strip()
        print(f"\nPerforming Benford Analysis for {symbol} on {exchange.capitalize()} from {start_datetime} to {end_datetime} in {term_days}-day terms.")
        
        if exchange in ["upbit", "bithumb"]:
            current_end = end_dt
        else:
            current_start = start_dt

        mac_values = {"first": [], "second": []}
        time_labels = []

        while True:
            if exchange in ["upbit", "bithumb"]:
                current_start = current_end - term_delta
            else:
                current_end = current_start + term_delta

            if (exchange in ["upbit", "bithumb"] and current_start < start_dt) or \
               (exchange == "binance" and current_end > end_dt):
                break

            try:
                if exchange == "binance":
                    df = fetch_data_binance(symbol, int(current_start.timestamp() * 1000), int(current_end.timestamp() * 1000))
                elif exchange == "upbit":
                    df = fetch_data_upbit(symbol, current_end)
                elif exchange == "bithumb":
                    df = fetch_data_bithumb(symbol, current_end)

                # 빈 데이터 확인
                if df.empty:
                    print(f"No data available for {symbol} from {current_start} to {current_end}.")
                    if exchange in ["upbit", "bithumb"]:
                        current_end -= timedelta(days=1)
                    else:
                        current_start += timedelta(days=1)
                    continue

                # 업비트 및 빗썸: 정렬 및 중복 제거
                if exchange in ["upbit", "bithumb"]:
                    df = df.sort_values(by="timestamp")
                    df = df.drop_duplicates(subset=["timestamp"], keep="first")  # 중복 제거

                # Benford 분석 수행
                analysis_results = calculate_price_change_rate_and_analyze(df, digit_type)

                for digit, (actual_frequencies, benford_dist) in analysis_results.items():
                    observed = actual_frequencies.values
                    expected = benford_dist.values
                    mad = np.mean(np.abs(expected - observed))
                    mac_values[digit].append(mad)

                    combined_data.append({
                        "symbol": symbol,
                        "start_date": current_start.strftime('%Y-%m-%d'),
                        "end_date": current_end.strftime('%Y-%m-%d'),
                        "digit_type": digit,
                        "mad": mad
                    })

                if current_start not in time_labels:
                    time_labels.append(current_start)

            except Exception as e:
                print(f"Error processing data for {symbol} from {current_start} to {current_end}: {e}")

            if exchange in ["upbit", "bithumb"]:
                current_end -= timedelta(days=1)
            else:
                current_start += timedelta(days=1)

        # 그래프 그리기
        if mac_values["first"] or mac_values["second"]:
            plot_mac_time_series(mac_values, time_labels, symbol, term_days, exchange, digit_type, start_datetime, end_datetime)

    # 최종 결과를 하나의 CSV로 저장
    if combined_data:
        combined_df = pd.DataFrame(combined_data)
        combined_csv_path = f"./crypto_data/Timeseries_data/MAC_result/{exchange.capitalize()}_{symbol}_MAC_Results_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}.csv"
        os.makedirs(os.path.dirname(combined_csv_path), exist_ok=True)
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

    perform_time_series_benford_analysis(exchange, symbols, start_datetime, end_datetime, term_days, digit_type)


if __name__ == "__main__":
    main()

