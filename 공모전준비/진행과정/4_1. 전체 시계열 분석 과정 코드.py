import pandas as pd
import os

#######################################################
# 현재 거래소, 간격, 대상 고정되어 있음 (바낸, 3일, TA)
#######################################################

def calculate_statistics(symbols, exchange, analysis_target, start_datetime, end_datetime, term_days):
    output_file = f"./crypto_data/Timeseries_data/MAC_result/{term_days}Day_{analysis_target}/전체정리파일_{exchange.capitalize()}_{analysis_target}_{term_days}day.csv"
    results_list = []

    for symbol in symbols:
        # 파일 경로 생성
        file_path = f"./crypto_data/Timeseries_data/MAC_result/{term_days}Day_{analysis_target}/{exchange.capitalize()}_{symbol}_{analysis_target}_MAC_Results_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day.csv"
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            print(f"파일을 찾을 수 없습니다: {file_path}")
            continue

        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 전체 기간 계산
        overall_period = f"{df['start_date'].min()} ~ {df['end_date'].max()}"

        # First와 Second로 그룹화하여 mean과 std 계산
        stats = df.groupby(['symbol', 'digit_type'])['mad'].agg(['mean', 'std']).reset_index()

        # Digit Type 값 변환 (첫 글자만 대문자)
        stats['digit_type'] = stats['digit_type'].str.capitalize()

        # 결과 포맷 변경
        stats = stats.rename(columns={'symbol': 'Symbol', 'digit_type': 'Digit Type', 'mean': 'Mean', 'std': 'Std Dev'})

        # 정리된 데이터프레임
        stats['전체 기간'] = overall_period
        stats = stats[['Symbol', '전체 기간', 'Digit Type', 'Mean', 'Std Dev']]

        # 결과를 리스트에 추가
        results_list.append(stats)

    # 모든 결과를 하나의 데이터프레임으로 병합
    if results_list:
        new_results = pd.concat(results_list, ignore_index=True)

        # 기존 데이터 읽기
        if os.path.exists(output_file):
            existing_results = pd.read_csv(output_file)

            # 기존 데이터와 새 데이터 병합 (같은 심볼과 Type은 덮어쓰기)
            final_results = pd.concat([existing_results, new_results]).drop_duplicates(subset=['Symbol', 'Digit Type'], keep='last').reset_index(drop=True)
        else:
            final_results = new_results

        # CSV 파일로 저장
        final_results.to_csv(output_file, index=False)

        print(f"파일이 성공적으로 저장되었습니다: {output_file}")
    else:
        print("처리할 데이터가 없습니다.")

if __name__ == "__main__":
    # 사용자 입력 받기
    symbols = [symbol.strip().upper() for symbol in input("분석할 심볼들을 콤마로 구분하여 입력하세요 (예: BTCUSDT,ETHUSDT): ").split(',')]
    exchange = "Binance"
    #exchange = input("거래소 이름을 입력하세요 (예: Binance): ").strip().capitalize()
    analysis_target = "TA"
    #analysis_target = input("분석 대상을 입력하세요 (예: TA): ")
    start_datetime = "2024-07-01-00:00"
    end_datetime = "2025-01-01-00:00"
    term_days = "3"
    #term_days = int(input("기간(일)을 입력하세요 (예: 1): "))

    calculate_statistics(symbols, exchange, analysis_target, start_datetime, end_datetime, term_days)
