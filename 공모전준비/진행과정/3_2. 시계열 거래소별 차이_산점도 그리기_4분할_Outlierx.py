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
    file_path = f"./crypto_data/Timeseries_data/MAC_result/21_{term_days}Day_TA/전체정리파일_{exchange_name}_{analysis_target}_{term_days}day.csv"
    save_path = f"./crypto_data/Timeseries_data/MAC_result/21_{term_days}Day_TA/전체 정리 그래프/"
    save_title = f"scatter_plots_{exchange_name}_{term_days}Day_{analysis_target}_no_outliers"
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

# IQR 기반으로 특이값 제거 함수 정의
def remove_outliers(x, y):
    # x, y 각각에 대해 IQR 계산
    q1_x, q3_x = np.percentile(x, [25, 75])
    iqr_x = q3_x - q1_x
    lower_bound_x = q1_x - 1.5 * iqr_x
    upper_bound_x = q3_x + 1.5 * iqr_x

    q1_y, q3_y = np.percentile(y, [25, 75])
    iqr_y = q3_y - q1_y
    lower_bound_y = q1_y - 1.5 * iqr_y
    upper_bound_y = q3_y + 1.5 * iqr_y

    # x와 y 모두 특이값 범위 안에 있는 데이터만 남김
    mask = (x >= lower_bound_x) & (x <= upper_bound_x) & (y >= lower_bound_y) & (y <= upper_bound_y)
    return x[mask], y[mask], [symbol for i, symbol in enumerate(symbol_labels) if mask[i]]

# 특이값 제거 적용
x, y, symbol_labels = remove_outliers(x, y)

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
for xi, yi, label in zip(x, y, symbol_labels):
    group = symbol_groups[label]
    groups[group].append((xi, yi, label))

# x축, y축 범위 설정
x_min, x_max = x.min() - 0.05 * abs(x.min()), x.max() + 0.05 * abs(x.max())
# y_min, y_max = y.min() - 0.05 * abs(y.min()), y.max() + 0.05 * abs(y.max())
y_min, y_max = y.min() - 0.05 * abs(y.min()), 0.0125

# 4분할 그래프 설정
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.flatten()

for idx, (group_name, data) in enumerate(groups.items()):
    ax = axes[idx]

    # 그룹 데이터 플롯
    scatter = ax.scatter(
        [xi for xi, yi, label in data],
        [yi for xi, yi, label in data],
        color=group_colors[group_name],
        alpha=1,
        s=50,
        label=f"{group_name} ({len(data)})"
    )
    
    # 각 점에 라벨 표시
    for xi, yi, label in data:
        ax.text(
            xi + 0.00005, yi + 0.00005, label, fontsize=9, ha='center', va='center', alpha=1,
            bbox=dict(facecolor='white', alpha=0, edgecolor='none')
        )

    # 원점(평균) 표시
    ax.axhline(y=y_origin, color='red', linestyle='--', linewidth=0.7, label=f"Y Mean ({y_origin:.5f})")
    ax.axvline(x=x_origin, color='green', linestyle='--', linewidth=0.7, label=f"X Mean ({x_origin:.5f})")

     # 원점(평균) 표시 / 개별 거래소
    if analysis_type == "IE":  # IE(개별 거래소)일 때만 실행
        ax.axhline(y=0.012, color='#404040', linestyle='-', linewidth=1, alpha=1, label="SD Marginal Conformity")
        ax.axvline(x=0.015, color='#bababa', linestyle='-', linewidth=1, alpha=1, label="FD Marginal Conformity")

    # 그래프 설정
    ax.set_title(f'{group_name} Group')
    ax.set_xlabel('First Digit Mean')
    ax.set_ylabel('Second Digit Mean')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper left')

# 레이아웃 조정 및 저장
plt.tight_layout()
output_path = os.path.join(save_path, f"{save_title}.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"그래프가 저장되었습니다: {output_path}")
