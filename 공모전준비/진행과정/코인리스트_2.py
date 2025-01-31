import requests
import pandas as pd

# CSV 파일 경로 설정
file_path = './BNCSV.csv'  # 여기에 CSV 파일 경로를 입력하세요

# CSV 파일 읽기
coin_df = pd.read_csv(file_path, encoding='utf-8-sig')

# 바이낸스 API 엔드포인트 (모든 코인의 24시간 거래 정보 가져오기)
url = "https://api.binance.com/api/v3/ticker/24hr"

# API 요청하여 데이터 가져오기
response = requests.get(url)
data = response.json()

# 거래대금(거래량 × 가격) 정보를 저장할 딕셔너리
trade_value_dict = {}

for item in data:
    symbol = item['symbol']
    quote_volume = float(item['quoteVolume'])  # 24시간 거래량
    last_price = float(item['lastPrice'])  # 마지막 가격
    trade_value = quote_volume * last_price  # 거래대금 계산
    trade_value_dict[symbol] = trade_value

# CSV 파일에서 심볼 컬럼을 기반으로 거래대금 추가
coin_df['24h Trade Value (USDT)'] = coin_df['심볼'].map(trade_value_dict)

# 거래대금 기준으로 내림차순 정렬
sorted_coin_df = coin_df.sort_values(by='24h Trade Value (USDT)', ascending=False)

# 정렬된 데이터를 새로운 CSV 파일로 저장
sorted_file_path = "./sorted_coins_by_trade_value.csv"
sorted_coin_df.to_csv(sorted_file_path, index=False, encoding='utf-8-sig')

# 결과 파일 경로 출력
print(f"정렬된 CSV 파일 저장 완료: {sorted_file_path}")
