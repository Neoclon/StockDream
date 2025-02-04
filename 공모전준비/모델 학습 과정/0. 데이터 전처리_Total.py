import os
import pandas as pd
import ast
from scipy.stats import entropy

def calculate_entropy(frequencies):
    """ 주어진 frequency 리스트에서 엔트로피를 계산하는 함수 (log base 2 사용) """
    return entropy(frequencies, base=2)

def process_symbol_group(symbols, exchange, start_datetime, end_datetime, term_days, analysis_target):
    """
    여러 개의 심볼을 입력받아, 해당하는 CSV 파일에서 First/Second Digit Frequency 엔트로피를 계산하여 저장.
    
    Parameters:
        symbols (list): 처리할 심볼 목록
        exchange (str): 거래소 이름 (예: "binance")
        start_datetime (str): 시작 날짜
        end_datetime (str): 종료 날짜
        term_days (int): 분석 기간 (1일 단위 등)
        analysis_target (str): 분석 대상 (예: "TA")
        mad_folder (str): MAD 값이 저장된 CSV 파일의 폴더 경로
    """

    for symbol in symbols:
        # Actual Frequency 파일 경로
        actual_file = f"./crypto_data/TraingData/AF_CSV/훈련용 데이터_23_BN/{exchange.capitalize()}_{symbol}_{analysis_target}_Actual_Frequency_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day.csv"
        
        # MAD 값 파일 경로
        mad_file = f"./crypto_data/TraingData/CSV/훈련용 데이터_23_BN/{exchange.capitalize()}_{symbol}_{analysis_target}_MAC_Results_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day.csv"

        # 파일이 존재하는지 확인
        if not os.path.exists(actual_file):
            print(f"❌ 파일 없음: {actual_file}")
            continue
        if not os.path.exists(mad_file):
            print(f"⚠️ MAD 파일 없음: {mad_file} (MAD 값 없이 진행)")

        # Actual Frequency CSV 파일 로드
        df_actual = pd.read_csv(actual_file)

        # actual_frequencies가 문자열 리스트로 저장된 경우 변환
        df_actual["actual_frequencies"] = df_actual["actual_frequencies"].apply(ast.literal_eval)

        # 엔트로피 계산 및 새로운 컬럼 추가
        df_actual["entropy"] = df_actual["actual_frequencies"].apply(calculate_entropy)

        # 기본 정보 컬럼 정리
        df_actual = df_actual[["symbol", "start_date", "end_date", "digit_type", "actual_frequencies", "entropy"]]

        # MAD 값 CSV 파일 로드 (파일이 있으면 불러오기)
        if os.path.exists(mad_file):
            df_mad = pd.read_csv(mad_file)
            df_mad = df_mad[["symbol", "start_date", "end_date", "digit_type", "mad"]]

            # 실제 빈도 데이터와 MAD 데이터 병합
            df_final = pd.merge(df_actual, df_mad, on=["symbol", "start_date", "end_date", "digit_type"], how="left")
        else:
            df_actual["mad"] = None  # MAD 데이터가 없으면 NaN 값으로 설정
            df_final = df_actual

        # 컬럼 순서 정리
        df_final = df_final[["symbol", "start_date", "end_date", "digit_type", "actual_frequencies", "mad", "entropy"]]

        # 심볼별 개별 CSV 저장 경로
        output_path = f"./crypto_data/TraingData/Total_CSV/0.수집_분류전/{exchange.capitalize()}_{symbol}_{analysis_target}_Total_CSV_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day.csv"

        # 저장할 디렉토리가 없는 경우 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 변환된 데이터 저장
        df_final.to_csv(output_path, index=False)
        #print(f"✅ {symbol} 엔트로피 계산 완료! 저장된 파일: {output_path}")

def main():
    # 기본 설정값
    exchange = "binance"
    start_datetime = "2023-01-01-00:00"
    end_datetime = "2024-01-01-00:00"
    term_days = 1
    analysis_target = "TA"

    # 심볼 그룹 입력
    print("\n🎯 심볼 무리를 입력하세요. 쉼표로 심볼 구분, 세미콜론(;)으로 그룹 구분")
    print("예시: BTCUSDT,ETHUSDT;XRPUSDT,DOGEUSDT;SOLUSDT,ADAUSDT")
    symbol_groups_input = input("📝 심볼 무리 입력: ").strip()

    # 입력된 문자열을 그룹 단위로 변환
    symbol_groups = [group.strip().split(",") for group in symbol_groups_input.split(";")]

    print(f"\n🚀 총 {len(symbol_groups)}개의 심볼 무리가 입력되었습니다.")

    # 각 심볼 그룹을 순차적으로 처리
    for group_idx, symbols in enumerate(symbol_groups, start=1):
        #print(f"\n▶ 심볼 무리 {group_idx}/{len(symbol_groups)} 처리 중: {symbols}")
        process_symbol_group(symbols, exchange, start_datetime, end_datetime, term_days, analysis_target)
        #print(f"✅ 심볼 무리 {group_idx}/{len(symbol_groups)} 처리 완료!\n")

# 실행
if __name__ == "__main__":
    main()
    print("✅ 심볼 무리 전체 처리 완료!\n")
