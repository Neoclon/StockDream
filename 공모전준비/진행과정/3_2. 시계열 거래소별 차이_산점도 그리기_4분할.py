import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 데이터 읽기
file_path = "./crypto_data/TS_Difference/바낸 vs 업비트 1차 결과/MAC_DATA_누적 copy.csv"
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

# x축: Second Digit Mean, y축: First Digit Mean
x = np.array(means_second)
y = np.array(means_first)

# 원점 계산 (평균)
x_origin = np.mean(x)
y_origin = np.mean(y)

# 색상 분류: 시가총액 순서 기준 그룹화
colors = []
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
    ax.set_title(f'{group_name}: Second Digit vs First Digit')
    ax.set_xlabel('Second Digit Mean')
    ax.set_ylabel('First Digit Mean')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.grid(True)

# 레이아웃 조정
plt.tight_layout()
output_path = "./crypto_data/TS_Difference/바낸 vs 업비트 1차 결과/scatter_plots_by_group.png"
plt.savefig(output_path, dpi=300)
plt.show()
