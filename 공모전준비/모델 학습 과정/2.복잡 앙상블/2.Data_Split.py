# data_split_normalization.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# data_split_normalization.py

import pandas as pd

def split_data(labeled_csv):
    # 전체 데이터 불러오기 (라벨링된 데이터)
    df = pd.read_csv(labeled_csv, parse_dates=["start_date", "end_date"])
    
    # 심볼별로 정렬하고, 각 심볼 내에서 start_date 기준으로 정렬
    df = df.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
    
    # Test 데이터: start_date가 2024-07-01-00:00 이후인 모든 데이터
    test = df[df["start_date"] >= pd.Timestamp("2024-07-01-00:00")].copy()
    test = test.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
    
    # Train & Valid 데이터: start_date가 2024-07-01-00:00 이전인 데이터
    tv = df[df["start_date"] < pd.Timestamp("2024-07-01-00:00")].copy()
    tv = tv.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
    
    # 심볼별로 Train & Valid 데이터를 분할
    # 고유 심볼 리스트를 시간 순(또는 알파벳 순)으로 정렬한 후 80:20으로 분할합니다.
    unique_symbols = sorted(tv["symbol"].unique())
    n = len(unique_symbols)
    n_train = int(n * 0.8)
    train_symbols = unique_symbols[:n_train]
    valid_symbols = unique_symbols[n_train:]
    
    train = tv[tv["symbol"].isin(train_symbols)].copy()
    valid = tv[tv["symbol"].isin(valid_symbols)].copy()
    
    # 각 데이터셋 내부에서도 심볼별, 시간 순으로 정렬
    train = train.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
    valid = valid.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
    test = test.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
    
    return train, valid, test

def normalize_features(df, feature_columns):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler

def main():
    labeled_csv = "./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/labeled_data.csv"  # 전체 라벨링된 데이터 CSV 파일 경로
    train, valid, test = split_data(labeled_csv)
    
    # 정규화할 피처: mad, entropy, F1~F9, S0~S9
    all_features = ["mad", "entropy"] + [f"F{i+1}" for i in range(9)] + [f"S{i}" for i in range(10)]
    # 실제로 존재하는 컬럼만 사용 (이미 데이터에 F1~F9 등 있다면 그대로 사용)
    feature_columns = [col for col in all_features if col in train.columns]
    
    train, scaler = normalize_features(train, feature_columns)
    valid[feature_columns] = scaler.transform(valid[feature_columns])
    test[feature_columns] = scaler.transform(test[feature_columns])

    # 저장
    train.to_csv("./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/train_data.csv", index=False)
    valid.to_csv("./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/valid_data.csv", index=False)
    test.to_csv("./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/test_data.csv", index=False)
    
    print("Train data saved:", len(train), "samples, unique symbols:", train["symbol"].nunique())
    print("Valid data saved:", len(valid), "samples, unique symbols:", valid["symbol"].nunique())
    print("Test data saved:", len(test), "samples, unique symbols:", test["symbol"].nunique())

if __name__ == "__main__":
    main()
