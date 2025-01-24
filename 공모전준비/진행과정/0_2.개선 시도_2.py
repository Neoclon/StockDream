import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드를 비활성화하고 파일 기반 렌더링 사용
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import gc

#################################################
# 현재 날짜: 2021-01-01-00:00 부터 2022-01-01-00:00
# 현재 거래소: binance
# 현재 type: both
# 현재 target: TA
#################################################

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

def fetch_data_upbit(symbol, start_datetime, end_datetime, max_retries=11, delay=1.4):
    """
    Upbit 데이터를 가져오는 함수에 딜레이와 재시도 로직 추가
    """
    all_data = []
    current_end_time = end_datetime

    while current_end_time > start_datetime:
        headers = {"Accept": "application/json"}
        params = {
            "market": symbol,
            "to": current_end_time.strftime('%Y-%m-%dT%H:%M:%S'),
            "count": UPBIT_LIMIT,
        }

        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(EXCHANGES["upbit"], params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

                if not data:
                    print(f"No more data available for {symbol} up to {current_end_time}.")
                    return pd.DataFrame()  # 빈 DataFrame 반환

                all_data.extend(data)

                # 마지막 데이터의 타임스탬프 - 1초로 갱신
                last_timestamp = datetime.strptime(data[-1]['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S')
                current_end_time = last_timestamp - timedelta(seconds=1)
                break  # 성공하면 재시도 루프 종료

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Too Many Requests
                    retries += 1
                    # print(f"Rate limit hit. Retrying {retries}/{max_retries}...")
                    time.sleep(delay * retries)  # 지수 백오프
                else:
                    raise e

        else:
            print(f"Failed to fetch data for {symbol} after {max_retries} retries.")
            break

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

def fetch_data_bithumb(symbol, start_datetime, end_datetime, max_retries=5, delay=1):
    """
    Bithumb 데이터를 가져오는 함수에 딜레이와 재시도 로직 추가
    """
    all_data = []
    current_end_time = end_datetime

    while current_end_time > start_datetime:
        headers = {"Accept": "application/json"}
        params = {
            "market": symbol,
            "to": current_end_time.strftime('%Y-%m-%dT%H:%M:%S'),
            "count": BITTHUMB_LIMIT,
        }

        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(EXCHANGES["bithumb"], params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

                if not data:
                    print(f"No more data available for {symbol} up to {current_end_time}.")
                    return pd.DataFrame()  # 빈 DataFrame 반환

                all_data.extend(data)

                # 마지막 데이터의 타임스탬프 - 1초로 갱신
                last_timestamp = datetime.strptime(data[-1]['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S')
                current_end_time = last_timestamp - timedelta(seconds=1)
                break  # 성공하면 재시도 루프 종료

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Too Many Requests
                    retries += 1
                    # print(f"Rate limit hit for Bithumb. Retrying {retries}/{max_retries}...")
                    time.sleep(delay * retries)  # 지수 백오프
                else:
                    raise e

        else:
            print(f"Failed to fetch data for {symbol} after {max_retries} retries.")
            break

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
    data = data.copy()  # 원본 데이터를 복사
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
    data = data.copy()  # 원본 데이터를 복사
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

from concurrent.futures import ThreadPoolExecutor

def fetch_data_parallel_upbit_bithumb(exchange, symbol, start_datetime, end_datetime, fetch_function, interval_days=28):
    """
    업비트와 빗썸에 병렬 처리 방식으로 데이터를 수집 (최신 → 과거 방향으로 요청)
    
    Args:
        exchange (str): 거래소 이름 ("upbit", "bithumb")
        symbol (str): 거래 심볼
        start_datetime (datetime): 데이터 시작 시간
        end_datetime (datetime): 데이터 종료 시간
        fetch_function (callable): 데이터를 가져오는 함수 (fetch_data_upbit 또는 fetch_data_bithumb)
        interval_days (int): 병렬 처리 시 시간 범위 (기본 30일)
    
    Returns:
        pd.DataFrame: 병렬로 수집한 데이터
    """
    # 최신 → 과거로 시간 범위를 나눔
    time_ranges = []
    current_end = end_datetime
    while current_end > start_datetime:
        current_start = max(start_datetime, current_end - timedelta(days=interval_days))
        time_ranges.append((current_start, current_end))
        current_end = current_start - timedelta(seconds=1)  # 중복 방지를 위해 1초 빼기

    # 병렬 처리로 데이터 요청
    with ThreadPoolExecutor(max_workers = min(10, os.cpu_count() + 4)) as executor:
        results = executor.map(
            lambda time_range: fetch_function(symbol, time_range[0], time_range[1]),
            time_ranges
        )

    # 데이터 병합
    try:
        return pd.concat(results, ignore_index=True)
    except ValueError:
        print(f"No data returned for {symbol} on {exchange}")
        return pd.DataFrame()  # 빈 DataFrame 반환

def perform_time_series_benford_analysis(exchange, symbols, start_datetime, end_datetime, term_days, digit_type, analysis_target):
    start_dt, _ = datetime_to_timestamp(start_datetime)
    end_dt, _ = datetime_to_timestamp(end_datetime)

    combined_data = []  # 분석 결과 저장

    for symbol in symbols:
        symbol = symbol.strip()
        print(f"\nPerforming Benford Analysis for {symbol} on {exchange.capitalize()} - {analysis_target} from {start_datetime} to {end_datetime} in {term_days}-day sliding terms.")

        mac_values = {"first": [], "second": []}
        time_labels = []

        if exchange == "binance":
            # 바이낸스: 기존 방식 유지
            current_start = start_dt
            while current_start < end_dt:
                current_end = current_start + timedelta(days=term_days)
                try:
                    df = fetch_data_binance(symbol, int(current_start.timestamp() * 1000), int(current_end.timestamp() * 1000))
                    current_start += timedelta(days=term_days)
                except Exception as e:
                    print(f"Error fetching Binance data for {symbol}: {e}")
                    current_start += timedelta(days=term_days)
                    continue

        elif exchange in ["upbit", "bithumb"]:
            # 업비트 & 빗썸: 병렬 처리 방식 (최신 → 과거 데이터)
            fetch_function = fetch_data_upbit if exchange == "upbit" else fetch_data_bithumb
            try:
                full_data = fetch_data_parallel_upbit_bithumb(exchange, symbol, start_dt, end_dt, fetch_function, interval_days=28)
                if full_data.empty:
                    print(f"No data available for {symbol} on {exchange.capitalize()}. Skipping...")
                    continue

                # 데이터를 term_days 단위로 슬라이딩 윈도우
                current_end = end_dt
                while current_end > start_dt:
                    current_start = max(start_dt, current_end - timedelta(days=term_days))
                    df = full_data[(full_data['timestamp'] >= current_start) & (full_data['timestamp'] < current_end)]

                    if df.empty:
                        print(f"No data available for {symbol} from {current_start} to {current_end}.")
                        current_end -= timedelta(days=term_days)
                        continue

                    # 벤포드 분석 수행
                    analysis_results = calculate_benford_analysis(df, analysis_target, digit_type)

                    for digit in ["first", "second"]:
                        if digit in analysis_results:
                            actual_frequencies, benford_dist = analysis_results[digit]
                            observed = actual_frequencies.values
                            expected = benford_dist.values
                            mad = np.mean(np.abs(expected - observed))
                            mac_values[digit].append(mad)

                            # combined_data에 결과 추가
                            combined_data.append({
                                "symbol": symbol,
                                "start_date": current_start.strftime('%Y-%m-%d-%H:%M'),
                                "end_date": current_end.strftime('%Y-%m-%d-%H:%M'),
                                "digit_type": digit,
                                "mad": mad,
                                "target": analysis_target
                            })

                    time_labels.append(current_end)
                    current_end -= timedelta(days=term_days)

            except Exception as e:
                print(f"Error fetching {exchange.capitalize()} data for {symbol}: {e}")
                continue

        else:
            print(f"Unsupported exchange: {exchange}")
            continue

    # 결과 저장
    if combined_data:
        combined_df = pd.DataFrame(combined_data)
        for symbol in combined_df['symbol'].unique():
            symbol_df = combined_df[combined_df['symbol'] == symbol]
            combined_csv_path = f"./crypto_data/Timeseries_data/MAC_result/{exchange.capitalize()}_{symbol}_{analysis_target}_MAC_Results_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day.csv"
            os.makedirs(os.path.dirname(combined_csv_path), exist_ok=True)
            symbol_df.to_csv(combined_csv_path, index=False)
            print(f"Saved results for {symbol} to {combined_csv_path}")

def perform_analysis_for_symbol(exchange, symbol, start_datetime, end_datetime, term_days, digit_type, analysis_target):
    """Single symbol analysis function."""
    perform_time_series_benford_analysis(exchange, [symbol], start_datetime, end_datetime, term_days, digit_type, analysis_target)

from concurrent.futures import ThreadPoolExecutor

def main():
    # Fixed values
    exchange = "upbit"
    start_datetime = "2021-01-01-00:00"
    end_datetime = "2022-01-01-00:00"
    term_days = 1
    digit_type = "both"
    analysis_target = "TA"

    # Multiple symbol groups input
    print("심볼 무리를 입력하세요. 쉼표로 구분하고 각 무리는 세미콜론(;)으로 구분하세요.")
    print("예시: BTCUSDT,ETHUSDT;XRPUSDT,DOGEUSDT;SOLUSDT,ADAUSDT")
    symbol_groups_input = input("심볼 무리 입력: ").strip()

    # Split the input into groups
    symbol_groups = [
        group.strip().split(",") for group in symbol_groups_input.split(";")
    ]

    print(f"총 {len(symbol_groups)}개의 심볼 무리가 입력되었습니다.")
    
    # Process each group sequentially
    for group_idx, symbols in enumerate(symbol_groups, start=1):
        print(f"\n심볼 무리 {group_idx}/{len(symbol_groups)} 작업 시작: {symbols}")
        
        # Parallel processing with ThreadPoolExecutor
        max_workers = min(32, os.cpu_count() + 4)  # Limit thread count
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    perform_analysis_for_symbol,
                    exchange,
                    symbol.strip(),
                    start_datetime,
                    end_datetime,
                    term_days,
                    digit_type,
                    analysis_target
                )
                for symbol in symbols
            ]

            # Wait for all tasks to complete and handle any exceptions
            for future in futures:
                try:
                    future.result()  # Wait for task completion and handle exceptions
                except Exception as e:
                    print(f"Error processing symbol: {e}")
        
        print(f"심볼 무리 {group_idx}/{len(symbol_groups)} 작업 완료!")

    print("모든 심볼 무리에 대한 작업이 완료되었습니다!")

def notify_completion():
    import os
    import platform
    if platform.system() == "Darwin":  # macOS
        os.system('say "Debugging complete"')
    elif platform.system() == "Linux":
        os.system('notify-send "Debugging complete"')
    elif platform.system() == "Windows":
        os.system('msg * "Debugging complete"')

def send_imessage(phone_number, message):
    """Send an iMessage to a specified phone number."""
    apple_script = f'''
    tell application "Messages"
        set targetService to 1st service whose service type = iMessage
        set targetBuddy to buddy "{phone_number}" of targetService
        send "{message}" to targetBuddy
    end tell
    '''
    os.system(f"osascript -e '{apple_script}'")

if __name__ == "__main__":
    main()
    phone_number = "010-9465-3976"  # 본인 전화번호 입력
    message = "Debugging complete!"  # 알림 메시지 내용
    send_imessage(phone_number, message)
    print("Debugging complete!")
    notify_completion()