import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#############################################
# 파일 및 저장 경로 확인하자
#############################################

# 거래소 이름 입력받기
exchange_name = input("거래소 이름을 입력하세요 (예: Binance, Upbit): ").strip()
term_days = int(input("간격을 입력하세요 (e.g., 14): ").strip())
analysis_target = input("analysis target을 입력하세요 (TA/TV/VCR/PCR): ").strip().upper()

# (개별 거래소용) 파일 및 저장 경로 설정
file_path = f"./crypto_data/Timeseries_data/MAC_result/{term_days}Day_TA/전체정리파일_{exchange_name}_{analysis_target}_{term_days}day.csv"
save_path = f"./crypto_data/Timeseries_data/MAC_result/{term_days}Day_TA/전체 정리 그래프/"
save_title = f"Errorbar_scatter_plots_{exchange_name}_{term_days}Day_{analysis_target}"

# (개별 거래소용) 파일 및 저장 경로 설정
#file_path = f"./crypto_data/TS_Difference/{term_days}Day_{analysis_target}/{term_days}_{analysis_target}_MAC_Comparison_DATA_누적.csv"
#save_path = f"./crypto_data/TS_Difference/{term_days}Day_{analysis_target}/{term_days}Day_TA/전체 정리 그래프/"
#save_title = f"MAC_Comparison_Errorbar_scatter_plots_{term_days}Day_{analysis_target}"

# 디렉토리가 없으면 생성
os.makedirs(save_path, exist_ok=True)

# 데이터 읽기
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"지정된 경로에 파일이 없습니다: {file_path}")
    exit()

# Symbol별 평균 및 표준편차 데이터 추출 (시가총액 순서 반영)
symbols = df['Symbol'].unique()
means_first = []
means_second = []
stds_first = []
stds_second = []
symbol_labels = []

for symbol in symbols:
    symbol_data = df[df['Symbol'] == symbol]
    #mean_first = symbol_data[symbol_data['Type'] == 'first']['Mean']
    #std_first = symbol_data[symbol_data['Type'] == 'first']['Std']
    #mean_second = symbol_data[symbol_data['Type'] == 'second']['Mean']
    #std_second = symbol_data[symbol_data['Type'] == 'second']['Std']
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
    ax.set_title(f'{group_name}: {exchange_name}_ErrorBar_{term_days}Day_{analysis_target}')
    ax.set_xlabel('First Digit Mean')
    ax.set_ylabel('Second Digit Mean')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    #ax.grid(True)
    ax.legend()

# 레이아웃 조정
plt.tight_layout()

output_path = os.path.join(save_path, f"{save_title}.png")
plt.savefig(output_path, dpi=300)
plt.show()
