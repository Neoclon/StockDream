# data_split_normalization.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# data_split_normalization.py

import pandas as pd

def split_data(labeled_csv, valid_symbol_count=70):
    # 라벨링된 전체 데이터 로드 (예: 2020-01-01 ~ 2025-01-01)
    df = pd.read_csv(labeled_csv, parse_dates=["start_date", "end_date"])
    
    # 전체 데이터를 심볼별로 정렬 (각 심볼 내 시간 순으로 정렬)
    df = df.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
    
    # Test 데이터: start_date가 2024-07-01-00:00 이후인 모든 데이터
    test = df[df["start_date"] >= pd.Timestamp("2024-07-01-00:00")].copy()
    test = test.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
    
    # Train & Valid 데이터: start_date가 2024-07-01-00:00 이전인 모든 데이터
    tv = df[df["start_date"] < pd.Timestamp("2024-07-01-00:00")].copy()
    tv = tv.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
    
    # 고유 심볼 목록을 무작위로 섞은 후 Valid 심볼 수를 valid_symbol_count로 고정
    symbols = np.array(tv["symbol"].unique())
    np.random.shuffle(symbols)
    # Valid 심볼 수를 70개로 고정하고, Train 심볼은 나머지 모든 심볼
    valid_symbols = symbols[:valid_symbol_count]
    train_symbols = symbols[valid_symbol_count:]
    
    train = tv[tv["symbol"].isin(train_symbols)].copy()
    valid = tv[tv["symbol"].isin(valid_symbols)].copy()
    
    # 각 데이터셋 내부에서도 심볼별, 시간 순으로 정렬
    train = train.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
    valid = valid.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
    test = test.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
    
    print("Train data:", len(train), "samples,", "Unique symbols:", train["symbol"].nunique())
    
    return train, valid, test

def normalize_features(df, feature_columns):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler

def main():
    labeled_csv = "./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/labeled_data_1.csv"  # 전체 라벨링된 데이터 CSV 파일 경로
    train, valid, test = split_data(labeled_csv)
    
    # 정규화할 피처: mad, entropy, F1~F9, S0~S9
    all_features = ["mad", "entropy"] + [f"F{i+1}" for i in range(9)] + [f"S{i}" for i in range(10)]
    # 실제로 존재하는 컬럼만 사용 (이미 데이터에 F1~F9 등 있다면 그대로 사용)
    feature_columns = [col for col in all_features if col in train.columns]
    
    train, scaler = normalize_features(train, feature_columns)
    valid[feature_columns] = scaler.transform(valid[feature_columns])
    test[feature_columns] = scaler.transform(test[feature_columns])

    # 저장
    train.to_csv("./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/train_data_1.csv", index=False)
    valid.to_csv("./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/valid_data_1.csv", index=False)
    test.to_csv("./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/test_data_1.csv", index=False)
    
    print("Train data saved:", len(train), "samples, unique symbols:", train["symbol"].nunique())
    print("Valid data saved:", len(valid), "samples, unique symbols:", valid["symbol"].nunique())
    print("Test data saved:", len(test), "samples, unique symbols:", test["symbol"].nunique())

if __name__ == "__main__":
    main()
