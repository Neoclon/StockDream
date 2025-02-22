import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#############################################
# 파일 및 저장 경로 확인하자
#############################################

# 분석 유형 입력받기
analysis_type = input("분석 유형을 입력하세요 (IE: 개별 거래소, IC: 거래소 비교): ").strip().upper()
term_days = int(input("간격을 입력하세요 (e.g., 14): ").strip())
analysis_target = input("analysis target을 입력하세요 (TA/TV/VCR/PCR): ").strip().upper()

# 파일 및 저장 경로 설정
if analysis_type == "IE":  # 개별 거래소
    exchange_name = input("거래소 이름을 입력하세요 (예: Binance, Upbit): ").strip()
    file_path = f"./crypto_data/Timeseries_data/MAC_result/22_{term_days}Day_TA/전체정리파일_{exchange_name}_{analysis_target}_{term_days}day.csv"
    save_path = f"./crypto_data/Timeseries_data/MAC_result/22_{term_days}Day_TA/전체 정리 그래프/"
    save_title = f"Errorbar_scatter_plots_{exchange_name}_{term_days}Day_{analysis_target}"
elif analysis_type == "IC":  # 거래소 비교
    file_path = f"./crypto_data/TS_Difference/{term_days}Day_{analysis_target}/{term_days}Day_{analysis_target}_MAC_Comparison_DATA_누적.csv"
    save_path = f"./crypto_data/TS_Difference/{term_days}Day_{analysis_target}/전체 정리 그래프/"
    save_title = f"MAC_Comparison_Errorbar_scatter_plots_{term_days}Day_{analysis_target}"
else:
    print("잘못된 분석 유형입니다. IE 또는 IC 중 하나를 입력하세요.")
    exit()

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

    if 'Type' in symbol_data.columns:  # 'Type' 컬럼이 있는 경우
        mean_first = symbol_data[symbol_data['Type'] == 'first']['Mean']
        mean_second = symbol_data[symbol_data['Type'] == 'second']['Mean']
        std_first = symbol_data[symbol_data['Type'] == 'first']['Std']
        std_second = symbol_data[symbol_data['Type'] == 'second']['Std']
    elif 'Digit Type' in symbol_data.columns:  # 'Digit Type' 컬럼이 있는 경우
        mean_first = symbol_data[symbol_data['Digit Type'] == 'First']['Mean']
        mean_second = symbol_data[symbol_data['Digit Type'] == 'Second']['Mean']
        std_first = symbol_data[symbol_data['Digit Type'] == 'First']['Std Dev']
        std_second = symbol_data[symbol_data['Digit Type'] == 'Second']['Std Dev']
    else:
        raise ValueError("DataFrame에 'Type' 또는 'Digit Type' 컬럼이 존재하지 않습니다.")
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

# 그룹화 함수 (파일 순서 기반)
def assign_group_by_file_order(symbols):
    total_symbols = len(symbols)
    group_labels = pd.cut(
        range(total_symbols),
        bins=[-1, int(total_symbols * 0.15), int(total_symbols * 0.35),
              int(total_symbols * 0.60), total_symbols],
        labels=['Top 15%', '15~35%', '35~60%', '60~100%']
    )
    return {symbols[i]: group_labels[i] for i in range(total_symbols)}

# 그룹화 및 색상 설정
group_colors = {
    'Top 15%': '#e41a1c',
    '15~35%': '#377eb8',
    '35~60%': '#4daf4a',
    '60~100%': '#984ea3'
}

# 그룹 할당
symbol_groups = assign_group_by_file_order(symbol_labels)

# 그룹별 데이터 구성
groups = {group: [] for group in group_colors.keys()}
for xi, yi, std_xi, std_yi, label in zip(x, y, std_x, std_y, symbol_labels):
    group = symbol_groups[label]
    groups[group].append((xi, yi, std_xi, std_yi, label))

# x축, y축 범위 설정 (1%만큼 확장)
x_min, x_max = x.min() - 0.05 * x.min(), x.max() + 0.05 * x.max()
y_min, y_max = y.min() - 0.5 * y.min(), y.max() + 0.5 * y.max()

# 4분할 그래프 설정
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.flatten()

for idx, (group_name, data) in enumerate(groups.items()):
    ax = axes[idx]

    for xi, yi, std_xi, std_yi, label in data:
        ax.errorbar(xi, yi, yerr=std_yi, fmt='o', color=group_colors[group_name], capsize=5, alpha=1,
                    label=f"{group_name} ({len(data)})" if label == data[0][-1] else "")
        #ax.text(xi + 0.00035, yi + 0.00035, label, fontsize=10, ha='center', va='center', alpha=0.9, 
                #bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    # 원점 표시
    ax.axhline(y=y_origin, color='red', linestyle='--', linewidth=1, label=f'Y Mean ({y_origin:.4f})')
    ax.axvline(x=x_origin, color='green', linestyle='--', linewidth=1, label=f'X Mean ({x_origin:.4f})')

    # 원점(평균) 표시 / 개별 거래소
    if analysis_type == "IE":  # IE(개별 거래소)일 때만 실행
        ax.axhline(y=0.012, color='#404040', linestyle='-', linewidth=1, alpha=1, label="SD Marginal Conformity")
        ax.axvline(x=0.015, color='#bababa', linestyle='-', linewidth=1, alpha=1, label="FD Marginal Conformity")

    # 그래프 설정
    if analysis_type == "IE":  # 개별 거래소
        ax.set_title(f'{exchange_name}_{term_days}_{analysis_target} : {group_name}')
        ax.set_xlabel('First Digit MAC')
        ax.set_ylabel('Second Digit MAC')
    elif analysis_type == "IC":  # 거래소 비교
        ax.set_title(f'Binance vs Upbit_{term_days}Day_{analysis_target} : {group_name}')
        ax.set_xlabel('First Digit MAC Differ')
        ax.set_ylabel('Second Digit MAC Differ')
    else:
        raise ValueError("잘못된 분석 유형입니다. IE 또는 IC 중 하나를 입력하세요.")
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    #ax.grid(True)
    ax.legend(loc='upper left')

# 레이아웃 조정
plt.tight_layout()

output_path = os.path.join(save_path, f"{save_title}.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"그래프가 저장되었습니다: {output_path}")