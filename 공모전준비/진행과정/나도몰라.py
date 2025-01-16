import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import time

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

def fetch_data_upbit(symbol, start_datetime, end_datetime, max_retries=5, delay=1):
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

# 일봉 데이터로 변환 함수
def convert_to_daily(data):
    daily_data = data.resample('D', on='timestamp').agg({
        'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'volume': 'sum'
    }).dropna()
    return daily_data

# 데이터 시각화 함수
def plot_daily_open_close(data, symbol):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 일봉 시가와 종가를 막대로 그리기
    daily_prices = data[['open', 'close']]
    for date, row in daily_prices.iterrows():
        open_price = row['open']
        close_price = row['close']
        if pd.notnull(open_price) and pd.notnull(close_price):
            color = '#D55E00' if close_price > open_price else '#0072B2'  # 상승일은 빨강, 하락일은 파랑
            ax.plot(
                [date, date],
                [open_price, close_price],
                color=color,
                linewidth=3,  # 막대 굵기
                alpha=0.9
            )

    ax.set_title(f'Daily Open and Close Prices for {symbol}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid()
    plt.show()

# 데이터 가져오기 함수
def fetch_data(symbol, start_datetime, end_datetime, exchange):
    if exchange == "binance":
        start_timestamp = int(start_datetime.timestamp() * 1000)
        end_timestamp = int(end_datetime.timestamp() * 1000)
        return fetch_data_binance(symbol, start_timestamp, end_timestamp)
    elif exchange == "upbit":
        return fetch_data_upbit(symbol, start_datetime, end_datetime)
    elif exchange == "bithumb":
        return fetch_data_bithumb(symbol, start_datetime, end_datetime)
    else:
        raise ValueError("Unsupported exchange")

# 심볼 데이터 가져오고 시각화
def main():
    symbol = "krw-btc"  # 원하는 심볼 입력 (Binance: BTCUSDT, Upbit: KRW-BTC, Bithumb: BTC_KRW)
    start_date = "2025-01-01"  # 시작 날짜
    end_date = "2025-01-15"  # 종료 날짜
    exchange = "upbit"  # 거래소 선택: binance, upbit, bithumb

    # 타임스탬프 변환
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

    # 데이터 가져오기
    try:
        data = fetch_data(symbol, start_datetime, end_datetime, exchange)
        daily_data = convert_to_daily(data)

        # 시각화
        plot_daily_open_close(daily_data, symbol)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
