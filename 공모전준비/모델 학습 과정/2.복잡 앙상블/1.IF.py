# isolation_forest_labeling.py
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
        print(f"Error in safe_eval: {x} | {e}")
        return np.nan

def load_data(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not all_files:
        print("No CSV files found in folder:", folder_path)
        return pd.DataFrame()
    data_frames = []
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df.reset_index(drop=True, inplace=True)
        if "actual_frequencies" in df.columns:
            df["actual_frequencies"] = df["actual_frequencies"].apply(safe_eval)
        # 분리: digit_type에 따라 실제 Actual Frequency 컬럼 생성
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

def label_data_with_isolation_forest(df, contamination="auto"):
    # First과 Second 그룹을 각각 처리
    df_first = df[df["digit_type"]=="first"].copy()
    df_second = df[df["digit_type"]=="second"].copy()
    
    feature_columns_first = ["mad", "entropy"] + [f"F{i+1}" for i in range(9)]
    feature_columns_second = ["mad", "entropy"] + [f"S{i}" for i in range(10)]
    
    for col in feature_columns_first:
        if df_first[col].isna().any():
            df_first[col].fillna(df_first[col].median(), inplace=True)
    for col in feature_columns_second:
        if df_second[col].isna().any():
            df_second[col].fillna(df_second[col].median(), inplace=True)
    
    model_first = IsolationForest(n_estimators=50, contamination=contamination, n_jobs=-1, random_state=42)
    model_second = IsolationForest(n_estimators=50, contamination=contamination, n_jobs=-1, random_state=42)
    
    start = time.time()
    X_first = df_first[feature_columns_first].values
    df_first["anomaly_label"] = model_first.fit_predict(X_first)
    df_first["anomaly_score"] = model_first.decision_function(X_first)
    elapsed_first = time.time() - start
    print(f"First Isolation Forest done in {elapsed_first:.2f} seconds")
    
    start = time.time()
    X_second = df_second[feature_columns_second].values
    df_second["anomaly_label"] = model_second.fit_predict(X_second)
    df_second["anomaly_score"] = model_second.decision_function(X_second)
    elapsed_second = time.time() - start
    print(f"Second Isolation Forest done in {elapsed_second:.2f} seconds")
    
    # 정규화: anomaly_score를 min-max로 [0,1] 매핑 (높을수록 이상치 가능성 높음)
    min_first, max_first = df_first["anomaly_score"].min(), df_first["anomaly_score"].max()
    df_first["if_prob"] = 1 - (df_first["anomaly_score"] - min_first) / (max_first - min_first + 1e-8)
    
    min_second, max_second = df_second["anomaly_score"].min(), df_second["anomaly_score"].max()
    df_second["if_prob"] = 1 - (df_second["anomaly_score"] - min_second) / (max_second - min_second + 1e-8)
    
    # 라벨은 threshold 0.5 기준: if_prob > 0.5 → anomaly=1, else 0
    df_first["label"] = (df_first["if_prob"] > 0.25).astype(int)
    df_second["label"] = (df_second["if_prob"] > 0.25).astype(int)
    
    # 이상치 비율 출력
    ratio_first = (df_first["label"]==1).mean() * 100
    ratio_second = (df_second["label"]==1).mean() * 100
    print(f"First group anomaly ratio: {ratio_first:.2f}%")
    print(f"Second group anomaly ratio: {ratio_second:.2f}%")
    
    # Combine back
    df_labeled = pd.concat([df_first, df_second], ignore_index=True)
    models = {"first": model_first, "second": model_second}
    return df_labeled, models

def main():
    data_folder = "./crypto_data/TraingData/Total_CSV/1.2_BN_Train/"
    df = load_data(data_folder)
    if df.empty:
        print("No data loaded.")
        return
    print(f"Data loaded: {len(df)} samples")
    
    df_labeled, models = label_data_with_isolation_forest(df, contamination="auto")
    # 저장: 라벨링 결과 CSV와 모델 저장 (두 모델을 하나의 딕셔너리로)
    df_labeled.to_csv("./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/labeled_data_1.csv", index=False)
    joblib.dump(models, "./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/combined_isolation_forest_models_1.pkl")
    print("Labeled data and models saved.")

if __name__ == "__main__":
    main()
