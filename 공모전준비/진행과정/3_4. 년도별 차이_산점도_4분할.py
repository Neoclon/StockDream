import pandas as pd
import matplotlib.pyplot as plt
import os

#############################################
# 두 파일 경로를 입력받아야 함
#############################################
# 분석에 필요한 입력 정보
term_days = int(input("간격을 입력하세요 (e.g., 14): ").strip())
analysis_target = input("analysis target을 입력하세요 (TA/TV/VCR/PCR): ").strip().upper()

# 파일 경로 설정 (예시: 2021년도와 2024년도)
file_path_2021 = f"./crypto_data/Timeseries_data/MAC_result/21_{term_days}Day_{analysis_target}/전체정리파일_Upbit_{analysis_target}_{term_days}day.csv"
file_path_2024 = f"./crypto_data/Timeseries_data/MAC_result/24_{term_days}Day_{analysis_target}/전체정리파일_Upbit_{analysis_target}_{term_days}day.csv"
save_path = f"./comparison_results/"
os.makedirs(save_path, exist_ok=True)

# 데이터 읽기
try:
    df_2021 = pd.read_csv(file_path_2021)
    df_2024 = pd.read_csv(file_path_2024)
except FileNotFoundError as e:
    print(f"지정된 경로에 파일이 없습니다: {e}")
    exit()

#############################################
# 데이터 처리 및 그룹화
#############################################
# 심볼별 평균 데이터 계산
def calculate_means(df):
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

    return pd.DataFrame({
        'Symbol': symbol_labels,
        'Mean_First': means_first,
        'Mean_Second': means_second
    })

# 평균 데이터 계산
data_2021 = calculate_means(df_2021)
data_2024 = calculate_means(df_2024)

# 공통 심볼 필터링
common_symbols = set(data_2021['Symbol']).intersection(set(data_2024['Symbol']))
data_2021 = data_2021[data_2021['Symbol'].isin(common_symbols)]
data_2024 = data_2024[data_2024['Symbol'].isin(common_symbols)]

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

data_2021 = assign_group(data_2021)
data_2024 = assign_group(data_2024)

#############################################
# x, y 축 범위 동등하게 설정
#############################################
x_values = pd.concat([data_2021['Mean_First'], data_2024['Mean_First']])
y_values = pd.concat([data_2021['Mean_Second'], data_2024['Mean_Second']])

x_min, x_max = x_values.min() - 0.05 * abs(x_values.min()), x_values.max() + 0.05 * abs(x_values.max())
y_min, y_max = y_values.min() - 0.05 * abs(y_values.min()), y_values.max() + 0.05 * abs(y_values.max())

#############################################
# 평균선 계산
#############################################
mean_x_2021 = data_2021['Mean_First'].mean()
mean_y_2021 = data_2021['Mean_Second'].mean()

mean_x_2024 = data_2024['Mean_First'].mean()
mean_y_2024 = data_2024['Mean_Second'].mean()

#############################################
# 4분할 그래프 생성
#############################################

group_colors = {
    'Top 15%': '#e41a1c',
    '15~35%': '#377eb8',
    '35~60%': '#4daf4a',
    '60~100%': '#984ea3'
}

fig, axes = plt.subplots(2, 2, figsize=(16, 16))  # 2x2 서브플롯 생성
axes = axes.flatten()  # 서브플롯을 1차원 리스트로 변환

for i, (group, color) in enumerate(group_colors.items()):
    ax = axes[i]
    
    # 2021 데이터
    group_data_2021 = data_2021[data_2021['Group'] == group]
    ax.scatter(group_data_2021['Mean_First'], group_data_2021['Mean_Second'],
               color=color, alpha=1, s=50, label=f'2021 {group}')
    #for _, row in group_data_2021.iterrows():
        #ax.text(row['Mean_First'], row['Mean_Second'], row['Symbol'], fontsize=6, alpha=1, color='black')
    
    
    # 2024 데이터
    group_data_2024 = data_2024[data_2024['Group'] == group]
    ax.scatter(group_data_2024['Mean_First'], group_data_2024['Mean_Second'],
               color=color, alpha=0.3, s=50, edgecolor='black', label=f'2024 {group}')
   #for _, row in group_data_2024.iterrows():
        #ax.text(row['Mean_First'], row['Mean_Second'], row['Symbol'], fontsize=6, alpha=1, color='black')
    
    
    # 평균선 표시
    #ax.axvline(mean_x_2021, color='blue', linestyle='--', linewidth=1, label=f'2021 Mean ({mean_x_2021:.4f})')
    #ax.axhline(mean_y_2021, color='blue', linestyle='--', linewidth=1, label=f'2021 Mean ({mean_y_2021:.4f})')
    #ax.axvline(mean_x_2024, color='orange', linestyle='--', linewidth=1, label=f'2024 Mean ({mean_x_2024:.4f})')
    #ax.axhline(mean_y_2024, color='orange', linestyle='--', linewidth=1, label=f'2024 Mean ({mean_y_2024:.4f})')
    ax.axhline(y=0.012, color='#404040', linestyle='-', linewidth=1, alpha=1, label="SD Marginal Conformity")
    ax.axvline(x=0.015, color='#bababa', linestyle='-', linewidth=1, alpha=1, label="FD Marginal Conformity")



    # 그래프 설정
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f'{group} Group')
    ax.set_xlabel('First Digit Mean')
    ax.set_ylabel('Second Digit Mean')
    ax.legend(loc='lower right')

plt.tight_layout()

# 그래프 저장
output_path = os.path.join(save_path, f"Comparison_4P_2021_vs_2024_{term_days}Day_{analysis_target}.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"4분할 그래프가 저장되었습니다: {output_path}")