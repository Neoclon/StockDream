import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 거래소 이름 입력받기
exchange = input("거래소 이름을 입력하세요 (예: Upbit, Binance): ").strip()

# 데이터 읽기
file_path = f"./crypto_data/Timeseries_data/MAC_result/1Day_TA/전체정리파일_{exchange}_TA_1day.csv"
save_path = "./crypto_data/Timeseries_data/MAC_result/1Day_TA/전체 정리 그래프/"
save_title = f"Scatter_Unity_1Day_TA_{exchange}"

# 디렉토리가 없으면 생성
os.makedirs(save_path, exist_ok=True)

# 데이터 읽기
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"지정된 경로에 파일이 없습니다: {file_path}")
    exit()

# Symbol별 평균 데이터 추출
symbols = df['Symbol'].unique()
means_first = []
means_second = []
symbol_labels = []

for symbol in symbols:
    symbol_data = df[df['Symbol'] == symbol]
    #mean_first = symbol_data[symbol_data['Type'] == 'first']['Mean']
    #mean_second = symbol_data[symbol_data['Type'] == 'second']['Mean']
    mean_first = symbol_data[symbol_data['Digit Type'] == 'First']['Mean']
    mean_second = symbol_data[symbol_data['Digit Type'] == 'Second']['Mean']
    means_first.append(mean_first)
    means_second.append(mean_second)
    symbol_labels.append(symbol)

# x축: Second Digit Mean, y축: First Digit Mean
x = np.array(means_second)
y = np.array(means_first)

# 원점 계산 (평균)
x_origin = np.mean(x)
y_origin = np.mean(y)

# 색상 분류: 시가총액 순서 기준 그룹화
colors = []
group_colors = ['#6a0dad', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
for i in range(len(symbols)):
    if i < 7:
        colors.append(group_colors[0])
    elif i < 21:
        colors.append(group_colors[1])
    elif i < 39:
        colors.append(group_colors[2])
    elif i < 63:
        colors.append(group_colors[3])
    else:
        colors.append(group_colors[4])

# 산점도 그리기
plt.figure(figsize=(12, 12))

for i, (xi, yi, color) in enumerate(zip(x, y, colors)):
    plt.scatter(xi, yi, color=color, alpha=0.9, s=100)
    plt.text(xi + 0.00005, yi - 0.00025, symbol_labels[i], fontsize=10, ha='center', va='center', alpha=0.9, 
             bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

# 원점 표시
plt.axhline(y=y_origin, color='red', linestyle='--', linewidth=1, label=f'Y Mean ({y_origin:.4f})')
plt.axvline(x=x_origin, color='green', linestyle='--', linewidth=1, label=f'X Mean ({x_origin:.4f})')

# 그래프 설정
plt.title(f'2D Scatter Plot : {exchange}')
plt.xlabel('Second Digit Mean')
plt.ylabel('First Digit Mean')
# plt.grid(True)
plt.legend()
plt.tight_layout()

# 그래프 저장
output_path = os.path.join(save_path, f"{save_title}.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"그래프가 저장되었습니다: {output_path}")