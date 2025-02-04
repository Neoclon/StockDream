import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import time

def safe_eval(x):
    try:
        return eval(x) if isinstance(x, str) else np.nan
    except Exception as e:
        print(f"❌ `safe_eval()` 변환 실패: {x} | 오류: {e}")
        return np.nan

def load_data(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not all_files:
        print("❌ 경로에 CSV 파일이 없습니다! 폴더 경로를 확인하세요.")
        return pd.DataFrame()
    
    data_frames = []
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df.reset_index(drop=True, inplace=True)

        # actual_frequencies 컬럼에 safe_eval 적용 (in-place 변환)
        if "actual_frequencies" in df.columns:
            df["actual_frequencies"] = df["actual_frequencies"].apply(safe_eval)
        
        # digit_type이 'first'인 경우 actual_frequencies 리스트 -> F1~F9
        if "digit_type" in df.columns:
            first_mask = df["digit_type"] == "first"
            if first_mask.any():
                first_features = pd.DataFrame(
                    df.loc[first_mask, "actual_frequencies"].tolist(),
                    columns=[f"F{i+1}" for i in range(9)],
                    index=df.loc[first_mask].index
                )
                df = df.join(first_features)

            # digit_type이 'second'인 경우 actual_frequencies 리스트 -> S0~S9
            second_mask = df["digit_type"] == "second"
            if second_mask.any():
                second_features = pd.DataFrame(
                    df.loc[second_mask, "actual_frequencies"].tolist(),
                    columns=[f"S{i}" for i in range(10)],
                    index=df.loc[second_mask].index
                )
                df = df.join(second_features)

        data_frames.append(df)
    
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        return pd.DataFrame()

def train_isolation_forest(df, contamination="auto"):
    """ NaN을 중앙값(median)으로 대체하는 버전 """
    # 사용할 컬럼 리스트
    feature_columns = ["mad", "entropy"] + [f"F{i+1}" for i in range(9)] + [f"S{i}" for i in range(10)]
    feature_columns = [col for col in feature_columns if col in df.columns]

    # df가 비어있는지 먼저 체크
    if df.empty:
        print("❌ DataFrame이 비어 있습니다. 학습을 진행할 수 없습니다.")
        return df, None
    
    # NaN을 가진 컬럼별로 중앙값으로 대체
    for col in feature_columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    # 다시 한 번 df가 비어있지 않은지 확인 (여기서는 행 제거를 안 하므로 비어있을 가능성은 낮음)
    if df.empty:
        print("❌ 중앙값 대체 이후, DataFrame이 비어 있습니다. 학습을 진행할 수 없습니다.")
        return df, None

    # Isolation Forest
    X = df[feature_columns].values
    model = IsolationForest(
        n_estimators=50, 
        contamination=contamination, 
        n_jobs=-1, 
        random_state=42
    )

    start_time = time.time()
    df["anomaly_score"] = model.fit_predict(X)
    end_time = time.time()

    print(f"✅ Isolation Forest 실행 완료! 걸린 시간: {end_time - start_time:.2f}초")
    return df, model

def main():
    data_folder = "./crypto_data/TraingData/Total_CSV/1.1_BN_Train/"
    output_file = "./crypto_data/TraingData/Total_CSV/1.1_BN_Train/Result/1.anomaly_results.csv"

    # 데이터 로드
    df = load_data(data_folder)
    if df.empty:
        print("❌ 불러온 데이터가 없습니다. 프로그램을 종료합니다.")
        return

    print(f"✅ 데이터 로드 완료: {len(df)}개 샘플")

    # Isolation Forest 학습 (NaN -> 중앙값 대체)
    df, model = train_isolation_forest(df, contamination="auto")
    if model is None:
        print("❌ 모델 학습이 완료되지 않아 저장할 데이터가 없습니다.")
        return

    # 결과 확인 및 저장
    anomaly_ratio = sum(df['anomaly_score'] == -1) / len(df) * 100
    print(f"✅ 이상 거래 탐지 완료 (이상치 비율: {anomaly_ratio:.2f}%)")

    df.to_csv(output_file, index=False)
    print(f"✅ 이상 탐지 결과 저장 완료: {output_file}")

if __name__ == "__main__":
    main()
