import os
import pandas as pd
import numpy as np


def prepare_and_save_training_data(base_folder, subfolder_prefix, window_size=5, exchange="Binance",
                                   output_first="first_training_data.npy", output_second="second_training_data.npy",
                                   processed_files_log="processed_files.txt"):
    """
    특정 조건의 CSV 파일에서 First Digit과 Second Digit 데이터를 분리하여 훈련 데이터를 저장
    :param base_folder: 기본 폴더 경로 (예: "./crypto_data/Timeseries_data/MAC_result/")
    :param subfolder_prefix: 하위 폴더 이름 구성 요소 (예: "21_1Day_TA")
    :param window_size: 슬라이딩 윈도우 크기
    :param exchange: "Binance" 또는 "Upbit"
    :param output_first: First Digit 데이터를 저장할 파일명
    :param output_second: Second Digit 데이터를 저장할 파일명
    :param processed_files_log: 처리된 파일 이름을 기록할 로그 파일
    """
    # 하위 폴더 경로 생성
    folder_path = os.path.join(base_folder, subfolder_prefix)

    # 기본 폴더가 없으면 오류 처리
    if not os.path.exists(folder_path):
        print(f"폴더 {folder_path}가 존재하지 않습니다. 확인해주세요.")
        return

    # 기존 처리된 파일 기록 불러오기
    if os.path.exists(processed_files_log):
        with open(processed_files_log, "r") as f:
            processed_files = set(f.read().splitlines())
    else:
        processed_files = set()

    first_digit_data = []
    second_digit_data = []

    # 기존 훈련 데이터 불러오기 (누적 저장용)
    if os.path.exists(output_first):
        first_digit_data = np.load(output_first).tolist()
    if os.path.exists(output_second):
        second_digit_data = np.load(output_second).tolist()

    # 폴더 내 파일 처리
    for file in os.listdir(folder_path):
        # 파일 형식 필터링: "Binance_" 또는 "Upbit_"
        if file.startswith(exchange + "_") and file.endswith("_1day.csv"):
            file_path = os.path.join(folder_path, file)

            # 중복 처리 방지
            if file in processed_files:
                print(f"파일 {file}은 이미 처리됨. 건너뜁니다.")
                continue

            # CSV 파일 읽기
            df = pd.read_csv(file_path)
            print(f"처리 중: {file}")

            # First Digit과 Second Digit 분리
            for digit_type in ["first", "second"]:
                filtered_df = df[df["digit_type"] == digit_type]
                for symbol in filtered_df["symbol"].unique():
                    symbol_data = filtered_df[filtered_df["symbol"] == symbol].sort_values(by="start_date")
                    mad_values = symbol_data["mad"].values

                    # 슬라이딩 윈도우 생성
                    for i in range(len(mad_values) - window_size):
                        if digit_type == "first":
                            first_digit_data.append(mad_values[i:i + window_size])
                        elif digit_type == "second":
                            second_digit_data.append(mad_values[i:i + window_size])

            # 처리된 파일 기록
            processed_files.add(file)

    # numpy array로 변환 및 저장
    first_digit_data = np.array(first_digit_data)
    second_digit_data = np.array(second_digit_data)

    np.save(output_first, first_digit_data)
    np.save(output_second, second_digit_data)

    # 처리된 파일 로그 업데이트
    with open(processed_files_log, "w") as f:
        f.write("\n".join(processed_files))

    print(f"First Digit 훈련 데이터 저장 완료: {output_first} (크기: {first_digit_data.shape})")
    print(f"Second Digit 훈련 데이터 저장 완료: {output_second} (크기: {second_digit_data.shape})")
    print(f"처리된 파일 로그 저장 완료: {processed_files_log}")


# 실행
if __name__ == "__main__":
    # 사용자 입력
    base_folder = "./crypto_data/Timeseries_data/MAC_result"
    subfolder_year = input("하위 폴더의 연도를 입력하세요 (예: 21): ").strip()
    subfolder_interval = input("하위 폴더의 간격을 입력하세요 (예: 1): ").strip()
    subfolder_target = input("하위 폴더의 타겟을 입력하세요 (예: TA): ").strip()
    exchange = input("분석할 거래소를 입력하세요 (Binance 또는 Upbit): ").strip().capitalize()

    # 하위 폴더 이름 조합
    subfolder_prefix = f"{subfolder_year}_{subfolder_interval}Day_{subfolder_target}"

    # 훈련 데이터 준비 및 저장
    prepare_and_save_training_data(
        base_folder=base_folder,
        subfolder_prefix=subfolder_prefix,
        window_size=5,
        exchange=exchange
    )
