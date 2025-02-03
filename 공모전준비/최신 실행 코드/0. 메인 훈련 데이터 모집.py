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
import aiohttp
import asyncio

#################################################
# 현재 날짜: 2023-01-01-00:00 부터 2024-01-01-00:00
# 현재 거래소: binance
# 현재 type: both
# 현재 Term Days: 1
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

async def fetch_data_binance(symbol, start_time, end_time):
    all_data = []
    current_start_time = start_time

    async with aiohttp.ClientSession() as session:
        while current_start_time < end_time:
            params = {
                "symbol": symbol,
                "interval": INTERVAL,
                "limit": BINANCE_LIMIT,
                "startTime": current_start_time,
                "endTime": end_time,
            }
            try:
                async with session.get(EXCHANGES["binance"], params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    if not data:
                        print(f"No more data available for {symbol} from {current_start_time} to {end_time}.")
                        break

                    all_data.extend(data)
                    last_timestamp = data[-1][0]
                    current_start_time = last_timestamp + 1

            except aiohttp.ClientError as e:
                print(f"Error fetching Binance data for {symbol}: {e}")
                break

    try:
        df = pd.DataFrame(all_data)
        df = df.iloc[:, :6]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        df["trade_amount"] = df["close"] * df["volume"]
        return df
    except Exception as e:
        print(f"Error processing Binance data for {symbol}: {e}")
        return pd.DataFrame()

async def fetch_data_upbit(symbol, start_datetime, end_datetime, max_retries=4, delay=1):
    all_data = []
    current_end_time = end_datetime

    async with aiohttp.ClientSession() as session:
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
                    async with session.get(EXCHANGES["upbit"], params=params, headers=headers) as response:
                        response.raise_for_status()
                        data = await response.json()

                        if not data:
                            print(f"No more data available for {symbol} up to {current_end_time}.")
                            return pd.DataFrame()

                        all_data.extend(data)
                        last_timestamp = datetime.strptime(data[-1]['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S')
                        current_end_time = last_timestamp - timedelta(seconds=1)
                        break

                except aiohttp.ClientError as e:
                    retries += 1
                    # print(f"Retrying ({retries}/{max_retries}) for Upbit: {e}")
                    await asyncio.sleep(delay * retries)
            else:
                print(f"Failed to fetch data for {symbol} after {max_retries} retries.")
                break

    try:
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
        return pd.DataFrame()

async def fetch_data_bithumb(symbol, start_datetime, end_datetime, max_retries=4, delay=1):
    all_data = []
    current_end_time = end_datetime

    async with aiohttp.ClientSession() as session:
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
                    async with session.get(EXCHANGES["bithumb"], params=params, headers=headers) as response:
                        response.raise_for_status()
                        data = await response.json()

                        if not data:
                            print(f"No more data available for {symbol} up to {current_end_time}.")
                            return pd.DataFrame()

                        all_data.extend(data)
                        last_timestamp = datetime.strptime(data[-1]['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S')
                        current_end_time = last_timestamp - timedelta(seconds=1)
                        break

                except aiohttp.ClientError as e:
                    retries += 1
                    print(f"Retrying ({retries}/{max_retries}) for Bithumb: {e}")
                    await asyncio.sleep(delay * retries)
            else:
                print(f"Failed to fetch data for {symbol} after {max_retries} retries.")
                break

    try:
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
        return pd.DataFrame()

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

def plot_mac_time_series(mac_values, time_labels, df, symbol, term_days, exchange, digit_type, analysis_target, start_datetime, end_datetime):
    """
    MAC 값과 날짜별 시가(open), 종가(close)를 별도 축으로 분리하여 표시.
    """
    # 그래프 전체 크기 설정
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})

    # 첫 번째 그래프 (MAC Values)
    if "first" in mac_values and mac_values["first"]:
        ax1.plot(
            time_labels,
            mac_values["first"],
            marker='o',
            markersize=2,
            linestyle='-',
            linewidth=0.7,
            color='#0072B2',
            alpha=0.7,
            label='First Digit MAC Values'
        )

    if "second" in mac_values and mac_values["second"]:
        ax1.plot(
            time_labels,
            mac_values["second"],
            marker='o',
            markersize=2,
            linestyle='-',
            linewidth=0.7,
            color='#E69F00',
            alpha=0.7,
            label='Second Digit MAC Values'
        )

    # 기준선 추가
    if digit_type == "first":
        ax1.axhline(y=0.006, color='green', linestyle='--', linewidth=0.5, label='Close Conformity')
        ax1.axhline(y=0.012, color='purple', linestyle='--', linewidth=0.5, label='Acceptable Conformity')
        ax1.axhline(y=0.015, color='red', linestyle='--', linewidth=0.5, label='Marginal Conformity')
    elif digit_type == "second":
        ax1.axhline(y=0.008, color='green', linestyle='--', linewidth=0.5, label='Close Conformity')
        ax1.axhline(y=0.010, color='purple', linestyle='--', linewidth=0.5, label='Acceptable Conformity')
        ax1.axhline(y=0.012, color='red', linestyle='--', linewidth=0.5, label='Marginal Conformity')
    elif digit_type == "both":
        ax1.axhline(y=0.012, color='purple', linestyle='--', linewidth=0.5, label='SD Marginal Conformity')
        ax1.axhline(y=0.015, color='red', linestyle='--', linewidth=0.5, label='FD Marginal Conformity')

    ax1.set_ylabel('MAC Values')
    ax1.grid(alpha=0.5)
    ax1.legend(loc='upper right')
    ax1.set_title(f'{exchange.capitalize()} - {symbol} - {analysis_target} - {term_days}-Day Term MAC and Price Time Series')

    # 기존 groupby를 resample로 대체
    daily_prices = df.resample('D', on='timestamp').agg({
        'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'volume': 'sum'
    }).dropna()  # 빈 데이터 제거

    for date, row in daily_prices.iterrows():
        open_price = row['open']
        close_price = row['close']
        if pd.notnull(open_price) and pd.notnull(close_price):
            if open_price == close_price:  # 시가와 종가가 같은 경우
                ax2.plot(
                    [date, date],
                    [open_price, close_price],
                    color='#D55E00',  # 빨간색 선
                    linewidth=1,  # 얇은 선
                    alpha=1
                )
            else:
                color = '#2ECC71' if close_price > open_price else '#E74C3C'  # 상승일은 초록, 하락일은 빨강. 바이낸스 차트 색상처럼
                ax2.plot(
                    [date, date],
                    [open_price, close_price],
                    color=color,
                    linewidth=3,  # 막대 굵기
                    alpha=1
                )

    ax2.set_ylabel('Price (Currency)')
    ax2.set_xlabel('Date')
    ax2.grid(alpha=0.5)

    # x축 날짜 포맷 조정
    fig.autofmt_xdate()

    # 그래프 저장
    graph_path = f"./crypto_data/TraingData/Graph/0.수집_분류전/{exchange.capitalize()}_{symbol}_{analysis_target}_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day_mac_and_price_timeseries_{digit_type}.png"
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()
    gc.collect()
    print(f"Saved MAC and Price Time Series graph to {graph_path}")

async def perform_time_series_benford_analysis(exchange, symbols, start_datetime, end_datetime, term_days, digit_type, analysis_target):
    start_dt, _ = datetime_to_timestamp(start_datetime)
    end_dt, _ = datetime_to_timestamp(end_datetime)
    term_delta = timedelta(days=term_days)

    combined_data = []  # MAC 저장용
    combined_actual_frequencies = []  # 새롭게 추가할 Actual Frequency 저장용
    full_data = []  # 전체 데이터 저장용

    for symbol in symbols:
        symbol = symbol.strip()
        print(f"\nPerforming Benford Analysis for {symbol} on {exchange.capitalize()} - {analysis_target} from {start_datetime} to {end_datetime} in {term_days}-day sliding terms.")

        mac_values = {"first": [], "second": []}
        time_labels = []

        # 시작 및 종료 시점 설정
        if exchange in ["upbit", "bithumb"]:
            current_end = end_dt
        else:  # Binance
            current_start = start_dt

        while True:
            if exchange in ["upbit", "bithumb"]:
                current_start = current_end - term_delta
            else:  # Binance
                current_end = current_start + term_delta

            # 종료 조건 확인
            if (exchange in ["upbit", "bithumb"] and current_start < start_dt) or \
               (exchange == "binance" and current_end > end_dt):
                break

            try:
                # 거래소별 데이터 가져오기 (비동기 호출)
                if exchange == "binance":
                    df = await fetch_data_binance(symbol, int(current_start.timestamp() * 1000), int(current_end.timestamp() * 1000))
                elif exchange == "upbit":
                    df = await fetch_data_upbit(symbol, current_start, current_end)
                elif exchange == "bithumb":
                    df = await fetch_data_bithumb(symbol, current_start, current_end)

                if df.empty:
                    print(f"No data available for {symbol} from {current_start} to {current_end}.")
                    if exchange in ["upbit", "bithumb"]:
                        current_end -= timedelta(days=1)
                    else:  # Binance
                        current_start += timedelta(days=1)
                    continue

                # 데이터 누적
                full_data.append(df)

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

                        # MAC 데이터 저장 (기존 방식 유지)
                        combined_data.append({
                            "symbol": symbol,
                            "start_date": current_start.strftime('%Y-%m-%d-%H:%M') if exchange != "upbit" else current_end.strftime('%Y-%m-%d-%H:%M'),
                            "end_date": current_end.strftime('%Y-%m-%d-%H:%M') if exchange != "upbit" else current_start.strftime('%Y-%m-%d-%H:%M'),
                            "digit_type": digit,
                            "mad": mad,
                            "target": analysis_target
                        })

                        # First & Second Digit의 Actual Frequency 저장 (새로운 추가 기능)
                        combined_actual_frequencies.append({
                            "symbol": symbol,
                            "start_date": current_start.strftime('%Y-%m-%d-%H:%M') if exchange != "upbit" else current_end.strftime('%Y-%m-%d-%H:%M'),
                            "end_date": current_end.strftime('%Y-%m-%d-%H:%M') if exchange != "upbit" else current_start.strftime('%Y-%m-%d-%H:%M'),
                            "actual_frequencies": actual_frequencies.tolist(),  # [0.301, 0.198, 0.101, ...] 형태로 저장
                            "digit_type": digit,  # "first" 또는 "second"
                            "target": analysis_target
                        })

                time_labels.append(current_start if exchange != "upbit" else current_end)

            except Exception as e:
                print(f"Error processing data for {symbol} from {current_start} to {current_end}: {e}")

            if exchange in ["upbit", "bithumb"]:
                current_end -= timedelta(days=1)  # 최신 데이터에서 과거로 이동
            else:  # Binance
                current_start += timedelta(days=1)  # 과거 데이터에서 최신으로 이동

        # MAC 시계열 그래프 생성
        if full_data and (mac_values["first"] or mac_values["second"]):
            full_df = pd.concat(full_data).drop_duplicates().reset_index(drop=True)
            plot_mac_time_series(
                mac_values=mac_values,
                time_labels=time_labels,
                df=full_df,  # 전체 데이터를 전달
                symbol=symbol,
                term_days=term_days,
                exchange=exchange,
                digit_type=digit_type,
                analysis_target=analysis_target,
                start_datetime=start_datetime,
                end_datetime=end_datetime
            )

    # 기존 MAC 결과 CSV 저장
    if combined_data:
        combined_df = pd.DataFrame(combined_data)
        for symbol in combined_df['symbol'].unique():
            symbol_df = combined_df[combined_df['symbol'] == symbol]
            combined_csv_path = f"./crypto_data/TraingData/CSV/0.수집_분류전/{exchange.capitalize()}_{symbol}_{analysis_target}_MAC_Results_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day.csv"
            os.makedirs(os.path.dirname(combined_csv_path), exist_ok=True)
            symbol_df.to_csv(combined_csv_path, index=False)
            print(f"Saved MAC results for {symbol} to {combined_csv_path}")

    # **새로운 Actual Frequency 결과 CSV 저장 (First & Second Digit)**
    if combined_actual_frequencies:
        combined_actual_df = pd.DataFrame(combined_actual_frequencies)
        for symbol in combined_actual_df['symbol'].unique():
            symbol_actual_df = combined_actual_df[combined_actual_df['symbol'] == symbol]
            actual_csv_path = f"./crypto_data/TraingData/AF_CSV/0.수집_분류전/{exchange.capitalize()}_{symbol}_{analysis_target}_Actual_Frequency_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day.csv"
            os.makedirs(os.path.dirname(actual_csv_path), exist_ok=True)
            symbol_actual_df.to_csv(actual_csv_path, index=False)
            print(f"Saved Actual Frequency results for {symbol} to {actual_csv_path}")

async def perform_analysis_for_symbol(exchange, symbol, start_datetime, end_datetime, term_days, digit_type, analysis_target):
    await perform_time_series_benford_analysis(exchange, [symbol], start_datetime, end_datetime, term_days, digit_type, analysis_target)

from concurrent.futures import ThreadPoolExecutor

async def main():
    # Fixed values
    exchange = "binance"
    start_datetime = "2020-01-01-00:00"
    end_datetime = "2021-01-01-00:00"
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
        
        # 비동기 작업 처리
        tasks = [
            perform_analysis_for_symbol(
                exchange,
                symbol.strip(),
                start_datetime,
                end_datetime,
                term_days,
                digit_type,
                analysis_target
            ) for symbol in symbols
        ]
        await asyncio.gather(*tasks)  # 모든 비동기 작업 동시 실행

        print(f"심볼 무리 {group_idx}/{len(symbol_groups)} 작업 완료!")

    print("모든 심볼 무리에 대한 작업이 완료되었습니다!")

if __name__ == "__main__":
    start=time.time()
    asyncio.run(main())
    elapsed_time = time.time() - start

    # 시, 분, 초로 변환
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("Debugging complete!")
    print(f"Elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")