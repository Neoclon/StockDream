import os
import pandas as pd

def split_and_save_csv(input_folder, output_folder_1, output_folder_2):
    """
    주어진 폴더의 CSV 파일을 6개월 단위로 분할하여 다른 폴더에 저장하는 함수.

    Parameters:
        input_folder (str): 원본 CSV 파일이 저장된 폴더
        output_folder_1 (str): 첫 번째 6개월 데이터를 저장할 폴더
        output_folder_2 (str): 두 번째 6개월 데이터를 저장할 폴더
    """

    # 저장할 폴더가 없으면 생성
    os.makedirs(output_folder_1, exist_ok=True)
    os.makedirs(output_folder_2, exist_ok=True)

    # 폴더 내 모든 CSV 파일 가져오기
    all_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    for file in all_files:
        file_path = os.path.join(input_folder, file)

        # CSV 파일 로드
        df = pd.read_csv(file_path)

        # 날짜 필터링을 위해 start_date 열을 datetime 형식으로 변환
        df["start_date"] = pd.to_datetime(df["start_date"])

        # 첫 번째 구간: 2024-01-01 ~ 2024-06-30
        df_1 = df[(df["start_date"] >= "2024-01-01") & (df["start_date"] <= "2024-06-30")]

        # 두 번째 구간: 2024-07-01 ~ 2025-01-01
        df_2 = df[(df["start_date"] >= "2024-07-01") & (df["start_date"] <= "2025-01-01")]

        # 기존 파일명에서 `exchange`, `symbol`, `analysis_target`, `term_days` 추출
        file_parts = file.split("_")
        if len(file_parts) < 7:
            print(f"❌ 파일명 형식이 다름: {file}")
            continue

        exchange = file_parts[0]
        symbol = file_parts[1]
        analysis_target = file_parts[2]
        term_days = file_parts[-1].replace("day.csv", "")  # 마지막 `1day.csv`에서 숫자만 추출

        # 첫 번째 6개월 데이터 저장
        if not df_1.empty:
            new_file_1 = f"{exchange}_{symbol}_{analysis_target}_Total_CSV_2024-01-01-00_00_to_2024-06-30-00_00_{term_days}day.csv"
            output_path_1 = os.path.join(output_folder_1, new_file_1)
            df_1.to_csv(output_path_1, index=False)
            print(f"✅ 저장 완료: {output_path_1}")

        # 두 번째 6개월 데이터 저장
        if not df_2.empty:
            new_file_2 = f"{exchange}_{symbol}_{analysis_target}_Total_CSV_2024-07-01-00_00_to_2025-01-01-00_00_{term_days}day.csv"
            output_path_2 = os.path.join(output_folder_2, new_file_2)
            df_2.to_csv(output_path_2, index=False)
            print(f"✅ 저장 완료: {output_path_2}")

def main():
    # 사용자 입력 받기
    input_folder = "./crypto_data/TraingData/Total_CSV/1.BN_24"
    output_folder_1 = "./crypto_data/TraingData/Total_CSV/1.BN_24/1.전반기"
    output_folder_2 = "./crypto_data/TraingData/Total_CSV/1.BN_24/2.전반기"

    # 실행
    split_and_save_csv(input_folder, output_folder_1, output_folder_2)

# 실행
if __name__ == "__main__":
    main()
