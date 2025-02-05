import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import time
import joblib

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
        
        # actual_frequencies 컬럼에 safe_eval 적용
        if "actual_frequencies" in df.columns:
            df["actual_frequencies"] = df["actual_frequencies"].apply(safe_eval)
        
        # digit_type에 따라 First와 Second 데이터 처리
        if "digit_type" in df.columns:
            first_mask = df["digit_type"] == "first"
            if first_mask.any():
                first_features = pd.DataFrame(
                    df.loc[first_mask, "actual_frequencies"].tolist(),
                    columns=[f"F{i+1}" for i in range(9)],
                    index=df.loc[first_mask].index
                )
                df = df.join(first_features)
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

def train_isolation_forest_separate(df, contamination="auto"):
    # First 데이터 처리
    df_first = df[df["digit_type"] == "first"].copy()
    # Second 데이터 처리
    df_second = df[df["digit_type"] == "second"].copy()
    
    # 사용할 피처 컬럼 정의
    feature_columns_first = ["mad", "entropy"] + [f"F{i+1}" for i in range(9)]
    feature_columns_second = ["mad", "entropy"] + [f"S{i}" for i in range(10)]
    
    # 결측치 처리: 각 컬럼별 중앙값 대체
    for col in feature_columns_first:
        if df_first[col].isna().any():
            df_first[col].fillna(df_first[col].median(), inplace=True)
    for col in feature_columns_second:
        if df_second[col].isna().any():
            df_second[col].fillna(df_second[col].median(), inplace=True)
    
    # First 모델 학습
    X_first = df_first[feature_columns_first].values
    model_first = IsolationForest(n_estimators=50, contamination=contamination, n_jobs=-1, random_state=42)
    start_time = time.time()
    df_first["anomaly_label"] = model_first.fit_predict(X_first)
    df_first["anomaly_score"] = model_first.decision_function(X_first)
    time_first = time.time() - start_time
    print(f"✅ First Isolation Forest 실행 완료! 걸린 시간: {time_first:.2f}초")
    
    min_score_first = df_first["anomaly_score"].min()
    max_score_first = df_first["anomaly_score"].max()
    df_first["if_prob"] = 1 - (df_first["anomaly_score"] - min_score_first) / (max_score_first - min_score_first + 1e-8)
    
    # Second 모델 학습
    X_second = df_second[feature_columns_second].values
    model_second = IsolationForest(n_estimators=50, contamination=contamination, n_jobs=-1, random_state=42)
    start_time = time.time()
    df_second["anomaly_label"] = model_second.fit_predict(X_second)
    df_second["anomaly_score"] = model_second.decision_function(X_second)
    time_second = time.time() - start_time
    print(f"✅ Second Isolation Forest 실행 완료! 걸린 시간: {time_second:.2f}초")
    
    min_score_second = df_second["anomaly_score"].min()
    max_score_second = df_second["anomaly_score"].max()
    df_second["if_prob"] = 1 - (df_second["anomaly_score"] - min_score_second) / (max_score_second - min_score_second + 1e-8)
    
    # First와 Second 그룹 각각의 이상치 비율 출력
    if not df_first.empty:
        anomaly_ratio_first = sum(df_first['anomaly_label'] == -1) / len(df_first) * 100
        print(f"✅ First 이상치 비율: {anomaly_ratio_first:.2f}%")
    if not df_second.empty:
        anomaly_ratio_second = sum(df_second['anomaly_label'] == -1) / len(df_second) * 100
        print(f"✅ Second 이상치 비율: {anomaly_ratio_second:.2f}%")

    # 두 데이터셋 결과를 합치는 옵션 (필요 시)
    df_all = pd.concat([df_first, df_second], ignore_index=True)
    return df_all, {"first": model_first, "second": model_second}

def main():
    data_folder = "./crypto_data/TraingData/Total_CSV/1.1_BN_Train/"
    
    # 데이터 로드
    df = load_data(data_folder)
    if df.empty:
        print("❌ 불러온 데이터가 없습니다. 프로그램을 종료합니다.")
        return
    print(f"✅ 데이터 로드 완료: {len(df)}개 샘플")
    
    # First, Second 데이터를 분리하여 각각 Isolation Forest 모델 학습
    df_all, models = train_isolation_forest_separate(df, contamination="auto")
    
    # 모델들을 하나의 딕셔너리에 합치기
    combined_models = {"first": models["first"], "second": models["second"]}
    
    # combined_models를 하나의 pkl 파일로 저장
    joblib.dump(combined_models, "combined_isolation_forest_models.pkl")
    print("✅ Combined Isolation Forest 모델 저장 완료: combined_isolation_forest_models.pkl")
    
    # 옵션: df_all을 CSV로 저장할 수 있습니다.
    # df_all.to_csv("anomaly_results_separate.csv", index=False)
    # print("✅ 이상 탐지 결과 CSV 저장 완료: anomaly_results_separate.csv")

if __name__ == "__main__":
    main()
