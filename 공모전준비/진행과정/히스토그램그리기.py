import pandas as pd
import matplotlib.pyplot as plt

# 데이터 읽기
file_path = "./crypto_data/TS_Difference/MAC_DATA_누적 copy.csv"
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

plt.figure(figsize=(15, 7))

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

# 그래프 출력
plt.show()
