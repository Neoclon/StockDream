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
save_title = f"Unitied_scatter_plots_by_group_{exchange_name}_{term_days}Day"

# (개별 거래소용) 파일 및 저장 경로 설정
#file_path = f"./crypto_data/TS_Difference/{term_days}Day_{analysis_target}/{term_days}_{analysis_target}_MAC_Comparison_DATA_누적.csv"
#save_path = f"./crypto_data/TS_Difference/{term_days}Day_{analysis_target}/{term_days}Day_TA/전체 정리 그래프/"
#save_title = f"MAC_Comparison_United_scatter_plots_{term_days}Day_{analysis_target}"

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
plt.title(f'Unity Scatter Plot : {exchange_name}_{term_days}Day_{analysis_target}')
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