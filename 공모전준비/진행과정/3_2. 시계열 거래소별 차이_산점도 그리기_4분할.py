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
    file_path = f"./crypto_data/Timeseries_data/MAC_result/{term_days}Day_TA/전체정리파일_{exchange_name}_{analysis_target}_{term_days}day copy.csv"
    save_path = f"./crypto_data/Timeseries_data/MAC_result/{term_days}Day_TA/전체 정리 그래프/"
    save_title = f"scatter_plots_{exchange_name}_{term_days}Day_{analysis_target}_STPTx"
elif analysis_type == "IC":  # 거래소 비교
    file_path = f"./crypto_data/TS_Difference/{term_days}Day_{analysis_target}/{term_days}Day_{analysis_target}_MAC_Comparison_DATA_누적.csv"
    save_path = f"./crypto_data/TS_Difference/{term_days}Day_{analysis_target}/전체 정리 그래프/"
    save_title = f"MAC_Comparison_scatter_plots_{term_days}Day_{analysis_target}"
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

# Symbol별 평균 데이터 추출
symbols = df['Symbol'].unique()
means_first = []
means_second = []
symbol_labels = []

for symbol in symbols:
    symbol_data = df[df['Symbol'] == symbol]
    if 'Type' in symbol_data.columns:  # 'Type' 컬럼이 있는 경우
        mean_first = symbol_data[symbol_data['Type'] == 'first']['Mean'].mean()
        mean_second = symbol_data[symbol_data['Type'] == 'second']['Mean'].mean()
    elif 'Digit Type' in symbol_data.columns:  # 'Digit Type' 컬럼이 있는 경우
        mean_first = symbol_data[symbol_data['Digit Type'] == 'First']['Mean'].mean()
        mean_second = symbol_data[symbol_data['Digit Type'] == 'Second']['Mean'].mean()
    else:
        raise ValueError("DataFrame에 'Type' 또는 'Digit Type' 컬럼이 존재하지 않습니다.")
    means_first.append(mean_first)
    means_second.append(mean_second)
    symbol_labels.append(symbol)

# x축: Second Digit Mean, y축: First Digit Mean
x = np.array(means_first)
y = np.array(means_second)

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

    # 그룹 데이터 플롯
    scatter = ax.scatter(
        [xi for xi, yi, label in data],
        [yi for xi, yi, label in data],
        color=group_colors[idx],
        alpha=0.7,
        s=50,
        label=group_name  # 그룹 범례 레이블 추가
    )
    
    # 각 점에 라벨 표시
    for xi, yi, label in data:
        ax.text(
            xi + 0.00035, yi + 0.00035, label, fontsize=10, ha='center', va='center', alpha=0.9,
            bbox=dict(facecolor='white', alpha=0, edgecolor='none')
        )

    # 원점(평균) 표시 / 거래소 간 비교
    ax.axhline(y=y_origin, color='red', linestyle='--', linewidth=0.7, label=f"Y Mean ({y_origin:.5f})")
    ax.axvline(x=x_origin, color='green', linestyle='--', linewidth=0.7, label=f"X Mean ({x_origin:.5f})")

    # 원점(평균) 표시 / 개별 거래소
    if analysis_type == "IE":  # IE(개별 거래소)일 때만 실행
        ax.axhline(y=0.012, color='#0072B2', linestyle='-', linewidth=0.7, alpha=0.5, label="SD Marginal Conformity")
        ax.axvline(x=0.015, color='#E69F00', linestyle='-', linewidth=0.7, alpha=0.5, label="FD Marginal Conformity")


    # 그래프 설정
    if analysis_type == "IE":  # 개별 거래소
        ax.set_title(f'{exchange_name}_{term_days}_{analysis_target} : {group_name}')
    elif analysis_type == "IC":  # 거래소 비교
        ax.set_title(f'Binance vs Upbit_{term_days}Day_{analysis_target} : {group_name}')
    else:
        raise ValueError("잘못된 분석 유형입니다. IE 또는 IC 중 하나를 입력하세요.")

    ax.set_xlabel('First Digit Differ Mean')
    ax.set_ylabel('Second Digit Differ Mean')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 범례 추가
    ax.legend(loc='lower right')  # 범례 위치와 크기 설정

# 레이아웃 조정 및 저장
plt.tight_layout()
output_path = os.path.join(save_path, f"{save_title}.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"그래프가 저장되었습니다: {output_path}")
