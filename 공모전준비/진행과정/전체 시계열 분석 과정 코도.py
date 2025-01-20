import pandas as pd
import os

def calculate_statistics(symbols, exchange, analysis_target, start_datetime, end_datetime, term_days):
    output_file = "./crypto_data/Timeseries_data/전체정리파일.csv"
    results_list = []

    for symbol in symbols:
        # 파일 경로 생성
        file_path = f"./crypto_data/Timeseries_data/MAC_result/{exchange.capitalize()}_{symbol}_{analysis_target}_MAC_Results_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day.csv"
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            print(f"파일을 찾을 수 없습니다: {file_path}")
            continue

        # CSV 파일 읽기
        df = pd.read_csv(file_path, header=None)

        # 컬럼 이름 설정
        df.columns = ['Symbol', 'Start', 'End', 'Type', 'Value', 'Category']

        # Value 열을 숫자로 변환 (숫자가 아닌 값은 NaN으로 처리)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

        # NaN 값 제거
        df = df.dropna(subset=['Value'])

        # First와 Second로 그룹화하여 mean과 std 계산
        stats = df.groupby(['Symbol', 'Type'])['Value'].agg(['mean', 'std']).reset_index()

        # 결과 포맷 변경
        stats = stats.rename(columns={'mean': 'Mean', 'std': 'Std'})

        # 정리된 데이터프레임
        stats['전체 기간'] = '전체 기간'
        stats = stats[['Symbol', '전체 기간', 'Type', 'Mean', 'Std']]

        # 결과를 리스트에 추가
        results_list.append(stats)

    # 모든 결과를 하나의 데이터프레임으로 병합
    if results_list:
        final_results = pd.concat(results_list, ignore_index=True)

        # CSV 파일로 저장 (누적 저장 방식)
        final_results.to_csv(output_file, index=False, header=not os.path.exists(output_file), mode='a')

        print(f"파일이 성공적으로 저장되었습니다: {output_file}")
    else:
        print("처리할 데이터가 없습니다.")

if __name__ == "__main__":
    # 사용자 입력 받기
    symbols = [symbol.strip().upper() for symbol in input("분석할 심볼들을 콤마로 구분하여 입력하세요 (예: BTCUSDT,ETHUSDT): ").split(',')]
    exchange = input("거래소 이름을 입력하세요 (예: Binance): ").strip().capitalize()
    analysis_target = input("분석 대상을 입력하세요 (예: TA): ")
    start_datetime = "2024-07-01-00:00"
    end_datetime = "2025-01-01-00:00"
    term_days = int(input("기간(일)을 입력하세요 (예: 1): "))

    calculate_statistics(symbols, exchange, analysis_target, start_datetime, end_datetime, term_days)
