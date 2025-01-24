import requests

def get_historical_market_cap(date):
    """
    특정 날짜의 비트코인 시가총액 및 가격 조회
    :param date: 날짜 (형식: "31-12-2021")
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/history"
    params = {"date": date, "localization": "false"}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        market_data = data.get("market_data", {})
        price = market_data.get("current_price", {}).get("usd")
        market_cap = market_data.get("market_cap", {}).get("usd")
        circulating_supply = market_data.get("circulating_supply")
        
        print(f"Date: {date}")
        print(f"Price: ${price}")
        print(f"Market Cap: ${market_cap}")
        print(f"Circulating Supply: {circulating_supply}")
    else:
        print("Error fetching historical data")
        print(f"Status Code: {response.status_code}, Response: {response.text}")

# 2021년 12월 31일 데이터 조회
get_historical_market_cap("31-12-2021")
