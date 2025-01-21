import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 거래소 이름 입력받기
exchange_name = input("거래소 이름을 입력하세요 (예: Binance, Upbit): ").strip()

# 파일 및 저장 경로 설정
file_path = f"./crypto_data/Timeseries_data/MAC_result/1Day_TA/전체정리파일_{exchange_name}_TA_1day.csv"
save_path = "./crypto_data/Timeseries_data/MAC_result/1Day_TA/전체 정리 그래프/"
save_title = f"scatter_plots_by_group_{exchange_name}"

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
    mean_first = symbol_data[symbol_data['Type'] == 'first']['Mean']
    mean_second = symbol_data[symbol_data['Type'] == 'second']['Mean']
    #mean_first = symbol_data[symbol_data['Digit Type'] == 'First']['Mean'].mean()
    #mean_second = symbol_data[symbol_data['Digit Type'] == 'Second']['Mean'].mean()
    means_first.append(mean_first)
    means_second.append(mean_second)
    symbol_labels.append(symbol)

# x축: Second Digit Mean, y축: First Digit Mean
x = np.array(means_second)
y = np.array(means_first)

# 원점 계산 (평균)
x_origin = np.mean(x)
y_origin = np.mean(y)

# 색상 분류 및 그룹화
group_colors = ['#6a0dad', '#1f77b4', '#2ca02c', '#ff7f0e']
groups = {"Top 7": [], "8-21": [], "22-39": [], "40-": []}
for i, (xi, yi) in enumerate(zip(x, y)):
    if i < 7:
        groups["Top 7"].append((xi, yi, symbol_labels[i]))
    elif i < 21:
        groups["8-21"].append((xi, yi, symbol_labels[i]))
    elif i < 39:
        groups["22-39"].append((xi, yi, symbol_labels[i]))
    else:
        groups["40-"].append((xi, yi, symbol_labels[i]))

# x축, y축 범위 설정
x_min, x_max = x.min() - 0.001, x.max() + 0.001
y_min, y_max = y.min() - 0.001, y.max() + 0.001

# 4분할 그래프 설정
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.flatten()

for idx, (group_name, data) in enumerate(groups.items()):
    ax = axes[idx]

    for xi, yi, label in data:
        ax.scatter(xi, yi, color=group_colors[idx], alpha=0.7, s=50)
        ax.text(xi + 0.00035, yi + 0.00035, label, fontsize=10, ha='center', va='center', alpha=0.9, 
                bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    # 원점 표시
    ax.axhline(y=y_origin, color='red', linestyle='--', linewidth=1)
    ax.axvline(x=x_origin, color='green', linestyle='--', linewidth=1)

    # 그래프 설정
    ax.set_title(f'{group_name}: ({exchange_name})')
    ax.set_xlabel('Second Digit Mean')
    ax.set_ylabel('First Digit Mean')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

# 레이아웃 조정 및 저장
plt.tight_layout()
output_path = os.path.join(save_path, f"{save_title}.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"그래프가 저장되었습니다: {output_path}")
