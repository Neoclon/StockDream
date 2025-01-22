import pandas as pd
import matplotlib.pyplot as plt
import os

# 데이터 읽기
file_path = "./crypto_data/Timeseries_data/MAC_result/1Day_TA/전체정리파일_Upbit_TA_1day.csv"
save_path = "./crypto_data/Timeseries_data/MAC_result/1Day_TA/전체 정리 그래프/"
save_title = "Histogram_1Day_TA_UP"

# 디렉토리가 없으면 생성
os.makedirs(save_path, exist_ok=True)

df = pd.read_csv(file_path)

# Symbol별 평균 데이터 추출
symbols = df['Symbol'].unique()
means_first = []
means_second = []
symbol_labels = []

for symbol in symbols:
    symbol_data = df[df['Symbol'] == symbol]
    mean_first = symbol_data[symbol_data['Digit Type'] == 'First']['Mean'].mean()
    mean_second = symbol_data[symbol_data['Digit Type'] == 'Second']['Mean'].mean()
    means_first.append(mean_first)
    means_second.append(mean_second)
    symbol_labels.append(symbol)

# 막대 그래프 그리기
x = range(len(symbols))
width = 0.35

plt.figure(figsize=(20, 9))

# First Digit 막대
plt.bar([i - width/2 for i in x], means_first, width=width, label='First Digit Means')

# Second Digit 막대
plt.bar([i + width/2 for i in x], means_second, width=width, label='Second Digit Means')

# 그래프 설정
plt.title('Mean Values of First and Second Digits by Symbol')
plt.xlabel('Symbols')
plt.ylabel('Mean')
plt.xticks(x, symbol_labels, rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(axis='y')

# 그래프 저장
output_path = os.path.join(save_path, f"{save_title}.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"그래프가 저장되었습니다: {output_path}")
