import requests
import pandas as pd
from datetime import datetime
import time

# 조사할 코인 리스트 (CoinGecko 기준 ID)
coin_list = [
    "bitcoin", "ethereum", "ripple", "dogecoin", "solana", "cardano", "tron", "chainlink", "stellar",
    "shiba-inu", "hedera-hashgraph", "polkadot", "bitcoin-cash", "near", "aptos", "aave", "vechain", "ethereum-classic",
    "algorand", "arbitrum", "cosmos", "stacks", "theta-token", "immutable-x", "the-graph", "sei", "the-sandbox",
    "eos", "tezos", "iota", "flow", "ethereum-name-service", "neo", "decentraland", "axie-infinity", "chiliz",
    "ecash", "mina-protocol", "kava", "1inch", "theta-fuel", "astar", "zilliqa", "0x", "just", "golem",
    "basic-attention-token", "celo", "ankr", "qtum", "siacoin", "gas", "stepn", "aelf", "threshold", "mask-network",
    "hive", "ontology", "swipe", "storj", "status", "lisk", "civic", "power-ledger", "everipedia", "iost",
    "stp-network", "strax", "ong", "pundi-x", "steem", "kyber-network-crystal", "ark"
]

# CoinGecko API URL
BASE_URL = "https://api.coingecko.com/api/v3/coins/"

def get_market_cap(coin, date):
    url = f"{BASE_URL}{coin}/history"
    params = {"date": datetime.strptime(date, '%Y-%m-%d').strftime('%d-%m-%Y'), "localization": "false"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        market_cap = data.get("market_data", {}).get("market_cap", {}).get("usd", None)
        return market_cap
    except Exception as e:
        print(f"Error fetching data for {coin}: {e}")
        return None

def fetch_market_caps(coin_list, date, chunk_size=10):
    result = []
    for i in range(0, len(coin_list), chunk_size):
        chunk = coin_list[i:i + chunk_size]
        for coin in chunk:
            market_cap = get_market_cap(coin, date)
            if market_cap is not None:
                result.append({"coin": coin, "market_cap_usd": market_cap})
            else:
                print(f"Skipping {coin} due to error.")
            time.sleep(2)  # 요청 간 딜레이 (2초)
        print("Pausing for 60 seconds to avoid rate limits...")
        time.sleep(60)  # 청크 처리 후 60초 대기
    return pd.DataFrame(result)

# 실행
date = "2021-12-31"  # 조회할 날짜
data = fetch_market_caps(coin_list, date)

# 결과 저장
data.to_csv(f"market_caps_{date}.csv", index=False)
print(f"Market cap data saved to market_caps_{date}.csv")
