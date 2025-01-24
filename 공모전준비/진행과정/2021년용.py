import requests
import pandas as pd
import time

# CryptoCompare API 키
API_KEY = ""
BASE_URL = "https://min-api.cryptocompare.com/data/v2/histoday"

coin_list = [
    "1INCH", "AAVE", "ADA", "ANKR", "ARK", "ATOM", "AXS", "BAT", "BCH", "BTC", 
    "CHZ", "CVC", "DOGE", "DOT", "ELF", "EOS", "ETC", "ETH", "FLOW", "GAS", "GLM", 
    "HBAR", "HIVE", "IOST", "IOTA", "IQ", "JST", "KAVA", "KNC", "LINK", "LSK", "MANA", 
    "NEAR", "NEO", "ONG", "ONT", "POWR", "PUNDIX", "QTUM", "SAND", "SC", "SNT", "STEEM", 
    "STORJ", "STPT", "STRAX", "STX", "SXP", "TFUEL", "THETA", "TRX", "VET", "XEC", "XLM", 
    "XRP", "XTZ", "ZIL", "ZRX"
]


def fetch_crypto_data(coin_list, date):
    """
    특정 날짜 기준으로 CryptoCompare에서 시가총액 데이터 가져오기
    """
    timestamp = int(pd.Timestamp(date).timestamp())
    result = []
    
    for coin in coin_list:
        try:
            params = {
                "fsym": coin,
                "tsym": "USD",
                "toTs": timestamp,
                "limit": 1,
                "api_key": API_KEY,
            }
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # 시가총액 데이터 추가
            market_data = data.get("Data", {}).get("Data", [])
            if market_data:
                close_price = market_data[-1]["close"]
                market_cap = market_data[-1]["volumeto"]  # CryptoCompare에서 대략적 시총 추정
                result.append({"coin": coin, "price": close_price, "market_cap": market_cap})
            else:
                print(f"No data for {coin} on {date}")

        except Exception as e:
            print(f"Error fetching data for {coin}: {e}")
        time.sleep(1)  # 요청 간 간격 추가

    return pd.DataFrame(result)

# 실행
date = "2021-12-31"
data = fetch_crypto_data(coin_list, date)

# 결과 정렬 후 저장
if not data.empty:
    data = data.sort_values(by="market_cap", ascending=False)  # 시가총액 기준 정렬
    data.to_csv(f"crypto_compare_market_caps_{date}.csv", index=False)
    print(f"Market cap data saved to crypto_compare_market_caps_{date}.csv (sorted by market cap)")
else:
    print("No valid market cap data available.")
