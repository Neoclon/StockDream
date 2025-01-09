import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table

# CSV 파일 경로 설정
cryptos = ['btc', 'eth', 'xrp', 'doge']  # 암호화폐 리스트

# 첫 번째 자릿수 추출 함수
def first_digit_analysis(data, column):
    # 첫 자릿수 추출
    data['First_Digit'] = data[column].apply(lambda x: int(str(x)[0]) if pd.notnull(x) and str(x)[0] != '0' else None)
    
    # 첫 자릿수 빈도 계산 (0인 값은 제외됨)
    digit_counts = data['First_Digit'].value_counts().sort_index()
    
    # 벤포드 분포
    benford_dist = [np.log10(1 + 1/d) for d in range(1, 10)]
    
    return digit_counts, benford_dist

# 벤포드 분석 시각화 함수 (그래프와 표를 따로 출력)
def plot_benford_analysis(digit_counts, benford_dist, title):
    # 그래프 출력
    plt.figure(figsize=(10, 6))
    plt.bar(digit_counts.index, digit_counts / digit_counts.sum(), alpha=0.7, label='Actual Distribution')
    plt.plot(range(1, 10), benford_dist, 'ro-', label='Benford Distribution')
    plt.title(title)
    plt.xlabel('First Digit')
    plt.ylabel('Frequency (Proportion)')
    plt.xticks(range(1, 10))
    plt.legend()
    plt.show()

    # Actual Frequency의 비중 계산 (비율)
    actual_frequencies = digit_counts / digit_counts.sum()

    # 소수점 4자리로 반올림
    actual_frequencies = actual_frequencies.round(4)
    benford_dist = [round(f, 4) for f in benford_dist]
    
    # 표로 결과 준비
    benford_table = pd.DataFrame({
        'First Digit': range(1, 10),
        'Actual Frequency (Proportion)': [actual_frequencies.get(i, 0) for i in range(1, 10)],
        'Benford Frequency': benford_dist
    })

    # 표 출력
    plt.figure(figsize=(8, 4))
    table_ax = plt.gca()
    table_ax.axis('off')
    table_plot = table(table_ax, benford_table, loc='center', colWidths=[0.2, 0.3, 0.3])

    # 표 스타일링 (배경 색상 등)
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.scale(1.2, 1.2)  # 크기 조정

    # 표 이미지 저장
    plt.show()


# 각 암호화폐 데이터에 대한 벤포드 분석
for crypto in cryptos:
    # 데이터 읽기
    file_name = f"{crypto}_data_2024.csv"
    data = pd.read_csv(file_name)
    
    # 가격과 거래량에 대한 첫 자릿수 분석
    # price_counts, benford_price_dist = first_digit_analysis(data, 'Close')
    volume_counts, benford_volume_dist = first_digit_analysis(data, 'Volume')

    # 가격 데이터에 대한 분석 시각화
    # plot_benford_analysis(price_counts, benford_price_dist, f"{crypto.capitalize()} Price - Benford's Law")
    
    # 거래량 데이터에 대한 분석 시각화
    plot_benford_analysis(volume_counts, benford_volume_dist, f"{crypto.capitalize()} Volume - Benford's Law (Excluding 0s)")

   
