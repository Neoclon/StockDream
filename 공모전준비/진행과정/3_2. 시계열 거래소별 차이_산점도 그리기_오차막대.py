import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 데이터 읽기
file_path = "./crypto_data/TS_Difference/바낸 vs 업비트 1차 결과/MAC_DATA_누적 copy.csv"
df = pd.read_csv(file_path)

# Symbol별 평균 및 표준편차 데이터 추출 (시가총액 순서 반영)
symbols = df['Symbol'].unique()
means_first = []
means_second = []
stds_first = []
stds_second = []
symbol_labels = []

for symbol in symbols:
    symbol_data = df[df['Symbol'] == symbol]
    mean_first = symbol_data[symbol_data['Digit Type'] == 'First']['Mean'].mean()
    std_first = symbol_data[symbol_data['Digit Type'] == 'First']['Std Dev'].mean()
    mean_second = symbol_data[symbol_data['Digit Type'] == 'Second']['Mean'].mean()
    std_second = symbol_data[symbol_data['Digit Type'] == 'Second']['Std Dev'].mean()
    means_first.append(mean_first)
    means_second.append(mean_second)
    stds_first.append(std_first)
    stds_second.append(std_second)
    symbol_labels.append(symbol)

# x축: First Digit Mean, y축: Second Digit Mean
x = np.array(means_first)
y = np.array(means_second)
std_x = np.array(stds_first)
std_y = np.array(stds_second)

# 95% 신뢰구간으로 표준편차 조정
#confidence_factor = 1.96  # 95% 신뢰구간
#std_x = std_x * confidence_factor

# 원점 계산 (평균)
x_origin = np.mean(x)
y_origin = np.mean(y)

# 색상 분류: 시가총액 순서 기준 그룹화
colors = []
group_colors = ['#6a0dad', '#1f77b4', '#2ca02c', '#ff7f0e']
groups = {"Top 7": [], "8-21": [], "22-39": [], "40-": []}
for i, (xi, yi, std_xi, std_yi) in enumerate(zip(x, y, std_x, std_y)):
    if i < 7:
        groups["Top 7"].append((xi, yi, std_xi, std_yi, symbol_labels[i]))
    elif i < 21:
        groups["8-21"].append((xi, yi, std_xi, std_yi, symbol_labels[i]))
    elif i < 39:
        groups["22-39"].append((xi, yi, std_xi, std_yi, symbol_labels[i]))
    else:
        groups["40-"].append((xi, yi, std_xi, std_yi, symbol_labels[i]))

# x축, y축 범위 설정
x_min, x_max = x.min() - 0.001, x.max() + 0.001
y_min, y_max = y.min() - 0.005, y.max() + 0.015

# 4분할 그래프 설정
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.flatten()

for idx, (group_name, data) in enumerate(groups.items()):
    ax = axes[idx]

    for xi, yi, std_xi, std_yi, label in data:
        ax.errorbar(xi, yi, yerr=std_yi, fmt='o', color=group_colors[idx], capsize=5, alpha=0.8,
                    label=group_name if label == data[0][-1] else "")
        #ax.text(xi + 0.00035, yi + 0.00035, label, fontsize=10, ha='center', va='center', alpha=0.9, 
                #bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    # 원점 표시
    ax.axhline(y=y_origin, color='red', linestyle='--', linewidth=1, label=f'Y Mean ({y_origin:.4f})')
    ax.axvline(x=x_origin, color='green', linestyle='--', linewidth=1, label=f'X Mean ({x_origin:.4f})')

    # 그래프 설정
    ax.set_title(f'{group_name}: Second Digit vs First Digit')
    ax.set_xlabel('First Digit Mean')
    ax.set_ylabel('Second Digit Mean')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    #ax.grid(True)
    ax.legend()

# 레이아웃 조정
plt.tight_layout()

output_path = "./crypto_data/TS_Difference/바낸 vs 업비트 1차 결과/scatter_plots_by_group_errorbar.png"
plt.savefig(output_path, dpi=300)
plt.show()
