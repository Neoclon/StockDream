import os
import pandas as pd
import numpy as np
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
        # actual_frequencies 컬럼에 safe_eval 적용
        if "actual_frequencies" in df.columns:
            df["actual_frequencies"] = df["actual_frequencies"].apply(safe_eval)
        # digit_type에 따라 추가 컬럼 생성
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

def test_isolation_forest(test_folder, model_path):
    # Load test data
    df = load_data(test_folder)
    if df.empty:
        print("No test data found.")
        return None
    print("Test data loaded:", len(df), "samples")
    
    # Split data by digit_type
    df_first = df[df["digit_type"]=="first"].copy()
    df_second = df[df["digit_type"]=="second"].copy()
    
    # Define feature columns for each group
    feature_columns_first = ["mad", "entropy"] + [f"F{i+1}" for i in range(9)]
    feature_columns_second = ["mad", "entropy"] + [f"S{i}" for i in range(10)]
    
    # Fill missing values with median for each group
    for col in feature_columns_first:
        if col not in df_first.columns:
            print(f"Column {col} not found in first group.")
        elif df_first[col].isna().any():
            df_first[col].fillna(df_first[col].median(), inplace=True)
    for col in feature_columns_second:
        if col not in df_second.columns:
            print(f"Column {col} not found in second group.")
        elif df_second[col].isna().any():
            df_second[col].fillna(df_second[col].median(), inplace=True)
    
    # Load combined Isolation Forest models (dictionary with keys "first" and "second")
    combined_models = joblib.load(model_path)
    model_first = combined_models["first"]
    model_second = combined_models["second"]
    
    # Predict for first group
    X_first = df_first[feature_columns_first].values
    scores_first = model_first.decision_function(X_first)
    min_score_first = scores_first.min()
    max_score_first = scores_first.max()
    df_first["if_prob"] = 1 - (scores_first - min_score_first) / (max_score_first - min_score_first + 1e-8)
    
    # Predict for second group
    X_second = df_second[feature_columns_second].values
    scores_second = model_second.decision_function(X_second)
    min_score_second = scores_second.min()
    max_score_second = scores_second.max()
    df_second["if_prob"] = 1 - (scores_second - min_score_second) / (max_score_second - min_score_second + 1e-8)
    
    # Ensure digit_type column exists (should already be present)
    df_first["digit_type"] = "first"
    df_second["digit_type"] = "second"
    
    # Combine results and select desired columns
    df_if = pd.concat(
        [
            df_first[["symbol", "start_date", "digit_type", "if_prob"]],
            df_second[["symbol", "start_date", "digit_type", "if_prob"]]
        ],
        ignore_index=True
    )
    return df_if

def main():
    test_folder = "./crypto_data/TraingData/Total_CSV/1.BN_24/2.후반기/"
    model_path = "./공모전준비/모델 학습 과정/학습 모델/YB_data/combined_isolation_forest_models.pkl"
    
    df_if = test_isolation_forest(test_folder, model_path)
    if df_if is not None:
        df_if.to_csv("./공모전준비/모델 학습 과정/결과/if_results.csv", index=False)
        print("Isolation Forest test results saved to if_results.csv")
        print(df_if.head())
        
        # -------------------------
        # 기존 전체 anomaly ratio (threshold=0.5)도 혹시 필요하다면 유지
        threshold = 0.5
        overall_anomaly_ratio = (df_if["if_prob"] > threshold).mean() * 100
        print(f"Overall Isolation Forest anomaly ratio (threshold=0.5): {overall_anomaly_ratio:.2f}%")
        # -------------------------

        # First와 Second를 다시 분리
        df_first = df_if[df_if["digit_type"] == "first"].copy()
        df_second = df_if[df_if["digit_type"] == "second"].copy()

        # -------------------------
        # 1. First/Second별로 평균, 표준편차 계산
        mean_first = df_first["if_prob"].mean()
        std_first  = df_first["if_prob"].std(ddof=1)  # sample std (ddof=1)

        mean_second = df_second["if_prob"].mean()
        std_second  = df_second["if_prob"].std(ddof=1)

        # 95% 신뢰구간 기준: mean + 1.96*std
        threshold_95_first = mean_first + 1.96 * std_first
        threshold_95_second = mean_second + 1.96 * std_second

        # 99% 신뢰구간 기준: mean + 2.58*std (통계적으로 2.575~2.58 정도 사용)
        threshold_99_first = mean_first + 2.58 * std_first
        threshold_99_second = mean_second + 2.58 * std_second

        # -------------------------
        # 2. 임계값 초과 비율(%) 계산
        #    - First
        anomaly_ratio_95_first = (df_first["if_prob"] > threshold_95_first).mean() * 100
        anomaly_ratio_99_first = (df_first["if_prob"] > threshold_99_first).mean() * 100
        #    - Second
        anomaly_ratio_95_second = (df_second["if_prob"] > threshold_95_second).mean() * 100
        anomaly_ratio_99_second = (df_second["if_prob"] > threshold_99_second).mean() * 100

        # -------------------------
        # 3. 임계값 초과 심볼 개수 (한번이라도 초과한 심볼)
        #    - First
        over_95_first_symbols = (df_first.groupby("symbol")["if_prob"].max() > threshold_95_first).sum()
        over_99_first_symbols = (df_first.groupby("symbol")["if_prob"].max() > threshold_99_first).sum()
        #    - Second
        over_95_second_symbols = (df_second.groupby("symbol")["if_prob"].max() > threshold_95_second).sum()
        over_99_second_symbols = (df_second.groupby("symbol")["if_prob"].max() > threshold_99_second).sum()

        # -------------------------

        N = 3  # 원하는 횟수 (2번 이상 초과하는 심볼만 카운트)

        # 1) 심볼별로 임계값을 초과한 일자(행)의 개수를 구함
        exceed_counts_first = df_first.groupby("symbol")["if_prob"].apply(lambda x: (x > 0.3).sum())
        exceed_counts_second = df_second.groupby("symbol")["if_prob"].apply(lambda x: (x > 0.5).sum())
        exceed_counts_first_99 = df_first.groupby("symbol")["if_prob"].apply(lambda x: (x > 0.5).sum())
        exceed_counts_second_99 = df_second.groupby("symbol")["if_prob"].apply(lambda x: (x > 0.5).sum())

        # 2) 그 횟수가 N번 이상이면 True
        symbols_exceed_n_times_first = exceed_counts_first >= N
        symbols_exceed_n_times_second = exceed_counts_second >= N
        symbols_exceed_n_times_first_99 = exceed_counts_first_99 >= N
        symbols_exceed_n_times_second_99 = exceed_counts_second_99 >= N

        # 3) True인 심볼이 몇 개인지
        count_symbols_exceed_n_times_first = symbols_exceed_n_times_first.sum()
        count_symbols_exceed_n_times_second= symbols_exceed_n_times_second.sum()
        count_symbols_exceed_n_times_first_99 = symbols_exceed_n_times_first_99.sum()
        count_symbols_exceed_n_times_second_99= symbols_exceed_n_times_second_99.sum()
        
        # 결과 출력
        print("\n=== First ===")
        print(f"Mean(if_prob) = {mean_first:.4f}, Std(if_prob) = {std_first:.4f}")
        print(f"95% 임계값 = {threshold_95_first:.4f}, 99% 임계값 = {threshold_99_first:.4f}")
        print(f"95% 임계값 초과 비율: {anomaly_ratio_95_first:.2f}%")
        print(f"99% 임계값 초과 비율: {anomaly_ratio_99_first:.2f}%")
        print(f"95% 임계값 초과 심볼 수: {over_95_first_symbols}개")
        print(f"99% 임계값 초과 심볼 수: {over_99_first_symbols}개")
        print(f"95% 임계값 초과(3회 이상) 심볼 수: {count_symbols_exceed_n_times_first}개")
        print(f"99% 임계값 초과(3회 이상) 심볼 수: {count_symbols_exceed_n_times_first_99}개")

        print("\n=== Second ===")
        print(f"Mean(if_prob) = {mean_second:.4f}, Std(if_prob) = {std_second:.4f}")
        print(f"95% 임계값 = {threshold_95_second:.4f}, 99% 임계값 = {threshold_99_second:.4f}")
        print(f"95% 임계값 초과 비율: {anomaly_ratio_95_second:.2f}%")
        print(f"99% 임계값 초과 비율: {anomaly_ratio_99_second:.2f}%")
        print(f"95% 임계값 초과 심볼 수: {over_95_second_symbols}개")
        print(f"99% 임계값 초과 심볼 수: {over_99_second_symbols}개")
        print(f"95% 임계값 초과(3회 이상) 심볼 수: {count_symbols_exceed_n_times_second}개")
        print(f"99% 임계값 초과(3회 이상) 심볼 수: {count_symbols_exceed_n_times_second_99}개")
        

if __name__ == "__main__":
    main()
