import requests
import pandas as pd

# 바이낸스 API 엔드포인트
url = "https://api.binance.com/api/v3/ticker/24hr"
response = requests.get(url)
data = response.json()

# 코인 심볼과 이름 저장을 위한 딕셔너리
coin_info = {}

# 데이터에서 코인 심볼 추출
for item in data:
    symbol = item['symbol']
    # USDT 마켓의 코인만 추출 (예: BTCUSDT, ETHUSDT 등)
    if symbol.endswith('USDT'):
        coin_symbol = symbol
        # 코인 이름은 심볼과 동일하게 설정 (필요에 따라 수정 가능)
        coin_name = coin_symbol.replace('USDT', '')
        coin_info[coin_name] = coin_symbol

# 딕셔너리를 데이터프레임으로 변환
df = pd.DataFrame(list(coin_info.items()), columns=["코인 이름", "심볼"])

# CSV 파일로 저장
df.to_csv("binance_coins.csv", index=False, encoding="utf-8-sig")
