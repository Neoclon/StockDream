import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#############################################
# 두 파일 경로를 입력받아야 함
#############################################
# 분석에 필요한 입력 정보
term_days = int(input("간격을 입력하세요 (e.g., 14): ").strip())
analysis_target = input("analysis target을 입력하세요 (TA/TV/VCR/PCR): ").strip().upper()
years_1 = int(input("비교할 첫번째 년도를 입력하세요(e.g., 21): ").strip())
years_2 = int(input("비교할 두번째 년도를 입력하세요(e.g., 22): ").strip())
exchange = input("원하는 거래소를 입력하세요(e.g., Binance/Upbit): ").strip().capitalize()

# 파일 경로 설정 (예시: 2021년도와 2024년도)
file_path_1 = f"./crypto_data/Timeseries_data/MAC_result/{years_1}_{term_days}Day_{analysis_target}/전체정리파일_{exchange}_{analysis_target}_{term_days}day.csv"
file_path_2 = f"./crypto_data/Timeseries_data/MAC_result/{years_2}_{term_days}Day_{analysis_target}/전체정리파일_{exchange}_{analysis_target}_{term_days}day.csv"
save_path = f"./comparison_results/"
save_title = f"Comparison_20{years_1}vs_20{years_2}_{exchange}_{term_days}Day_{analysis_target}"

# 디렉토리 생성
os.makedirs(save_path, exist_ok=True)

# 데이터 읽기
try:
    df_1 = pd.read_csv(file_path_1)
    df_2 = pd.read_csv(file_path_2)
except FileNotFoundError as e:
    print(f"지정된 경로에 파일이 없습니다: {e}")
    exit()

#############################################
# 데이터 처리 및 그룹화
#############################################
# 심볼별 평균 데이터 계산
def calculate_means(df, year):
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
            raise ValueError(f"DataFrame에 'Type' 또는 'Digit Type' 컬럼이 존재하지 않습니다. (연도: {year})")
        means_first.append(mean_first)
        means_second.append(mean_second)
        symbol_labels.append(symbol)

    return pd.DataFrame({
        'Symbol': symbol_labels,
        'Mean_First': means_first,
        'Mean_Second': means_second
    })

# 각 연도별 평균 계산
data_1= calculate_means(df_1, 2021)
data_2 = calculate_means(df_2, 2024)

# 공통 심볼 필터링
#common_symbols = set(data_1['Symbol']).intersection(set(data_2['Symbol']))
#data_1 = data_1[data_1['Symbol'].isin(common_symbols)]
#data_2 = data_2[data_2['Symbol'].isin(common_symbols)]

# 시가총액 순서에 따라 그룹화 (상위 15%, 15~35%, 35~60%, 60~100%)
def assign_group(df):
    total_symbols = len(df)
    df = df.reset_index(drop=True)  # 기존 순서 유지
    df['Group'] = pd.cut(
        df.index,
        bins=[-1, int(total_symbols * 0.15), int(total_symbols * 0.35),
              int(total_symbols * 0.60), total_symbols],
        labels=['Top 15%', '15~35%', '35~60%', '60~100%']
    )
    return df

data_1 = assign_group(data_1)
data_2 = assign_group(data_2)

#############################################
# 4분할 그래프 그리기
#############################################
plt.figure(figsize=(12, 12))

# 그룹별 색상 설정
group_colors = {
    'Top 15%': '#e41a1c',
    '15~35%': '#377eb8',
    '35~60%': '#4daf4a',
    '60~100%': '#984ea3'
}

for group, color in group_colors.items():
    # 첫번째 년도 데이터
    group_data_1 = data_1[data_1['Group'] == group]
    plt.scatter(group_data_1['Mean_First'], group_data_1['Mean_Second'],
                color=color, alpha=1, s=50, label=f'20{years_1} {group} ({len(group_data_1)})')
    #for _, row in group_data_2021.iterrows():
        #plt.text(row['Mean_First'], row['Mean_Second'], row['Symbol'], fontsize=9, alpha=0.7, color=color)

    # 2024 데이터
    group_data_2 = data_2[data_2['Group'] == group]
    plt.scatter(group_data_2['Mean_First'], group_data_2['Mean_Second'],
                color=color, alpha=0.3, s=50, edgecolor='black', label=f'20{years_2} {group} ({len(group_data_2)})')
    #for _, row in group_data_2024.iterrows():
        #plt.text(row['Mean_First'], row['Mean_Second'], row['Symbol'], fontsize=9, alpha=0.7, color=color)

# 그래프 설정
plt.axhline(y=0.012, color='#404040', linestyle='-', linewidth=1, alpha=1, label="SD Marginal Conformity")
plt.axvline(x=0.015, color='#bababa', linestyle='-', linewidth=1, alpha=1, label="FD Marginal Conformity")
plt.title(f'Comparison of {exchange}: 20{years_1} vs 20{years_2} ({term_days}Day_{analysis_target})')
plt.xlabel('First Digit Mean')
plt.ylabel('Second Digit Mean')
plt.legend(loc='upper left')
plt.tight_layout()

# 그래프 저장
output_path = os.path.join(save_path, f"{save_title}.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"4분할 그래프가 저장되었습니다: {output_path}")
