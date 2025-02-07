import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ✅ Mac GPU(MPS) 지원 여부 체크 & 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Metal API (Mac GPU)
    print("🚀 Using device: Mac GPU (Metal MPS)")
else:
    device = torch.device("cpu")  # MPS가 안되면 CPU 사용
    print("⚠️ MPS 지원되지 않음. CPU 사용")

# ✅ 안전한 eval 변환 함수 (오류 방지)
def safe_eval(x):
    try:
        return eval(x) if isinstance(x, str) else np.nan
    except Exception as e:
        print(f"❌ `safe_eval()` 변환 실패: {x} | 오류: {e}")
        return np.nan

# ✅ 데이터 로드 함수 (LSTM용)
def load_lstm_data(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    data_frames = []

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # ✅ 컬럼 확인 (디버깅)
        # print(f"📂 파일: {file} | 포함된 컬럼: {df.columns.tolist()}")

        # ✅ 날짜 변환 및 정렬
        df["start_date"] = pd.to_datetime(df["start_date"])
        df = df.sort_values(by=["symbol", "start_date"])  # 암호화폐별 시간순 정렬

        # ✅ actual_frequencies 변환
        if "actual_frequencies" in df.columns:
            df["actual_frequencies"] = df["actual_frequencies"].apply(safe_eval)

        # ✅ First Digit 변환 (F1~F9)
        first_mask = df["digit_type"] == "first"
        if first_mask.any():
            first_features = pd.DataFrame(
                df.loc[first_mask, "actual_frequencies"].tolist(),
                columns=[f"F{i+1}" for i in range(9)],
                index=df.loc[first_mask].index
            )
            df = df.join(first_features)

        # ✅ Second Digit 변환 (S0~S9)
        second_mask = df["digit_type"] == "second"
        if second_mask.any():
            second_features = pd.DataFrame(
                df.loc[second_mask, "actual_frequencies"].tolist(),
                columns=[f"S{i}" for i in range(10)],
                index=df.loc[second_mask].index
            )
            df = df.join(second_features)

        # ✅ Feature 컬럼 설정
        feature_cols = ["mad"] + [f"F{i+1}" for i in range(9)] + [f"S{i}" for i in range(10)]

        # ✅ 누락된 컬럼 확인
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 누락된 컬럼: {missing_cols} | 파일: {file}")

        # ✅ 최종 Feature 정리
        df = df[["symbol", "start_date"] + feature_cols]

        data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)

# ✅ LSTM 데이터셋 클래스 (오류 방지 코드 추가)
class CryptoTimeSeriesDataset(Dataset):
    def __init__(self, df, sequence_lengths=[15, 20, 25, 30]):
        self.sequence_lengths = sequence_lengths
        self.grouped_data = {symbol: df_group for symbol, df_group in df.groupby("symbol")}

        # ✅ 전체 시퀀스 데이터 저장
        self.X, self.y = [], []

        # ✅ 각 코인의 데이터를 시계열 형태로 변환
        for symbol, group in self.grouped_data.items():
            features = group.iloc[:, 2:].values  # symbol, date 제외
            num_features = features.shape[1]  # feature 개수

            # ✅ 모든 시퀀스 길이별로 확인
            for seq_length in self.sequence_lengths:
                for i in range(len(features) - seq_length):
                    seq = features[i:i + seq_length]  # (seq_length, num_features)

                    # ✅ 시퀀스 크기 체크 (정확한 shape 보장)
                    if seq.shape == (seq_length, num_features):
                        self.X.append(seq)
                        self.y.append(features[i + seq_length][-1])  # 마지막 값 예측

        # ✅ NumPy 배열 변환 (오류 방지)
        self.X = np.array(self.X, dtype=np.float32)  # (samples, seq_length, features)
        self.y = np.array(self.y, dtype=np.float32)  # (samples,)

        print(f"✅ 데이터셋 생성 완료! 총 샘플 수: {len(self.X)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# ✅ LSTM 모델 정의
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # 마지막 시퀀스 출력값
        out = self.fc(last_out)
        return self.sigmoid(out)  # 확률값 반환

# ✅ 모델 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# ✅ 실행
def main():
    data_folder = "./crypto_data/TraingData/Total_CSV/1.1_BN_Train/"
    df = load_lstm_data(data_folder)

    # ✅ Dataset & DataLoader 생성 (다양한 시퀀스 길이 적용)
    sequence_lengths = [15, 20, 25, 30]
    dataset = CryptoTimeSeriesDataset(df, sequence_lengths)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"✅ 총 {len(dataset)}개 시퀀스 데이터 생성 완료!")

    # ✅ LSTM 모델 초기화
    input_dim = 21  # mad (first + second), F1~F9, S0~S9 (총 21개 feature)
    hidden_dim = 64
    num_layers = 2
    output_dim = 1

    model = LSTMPredictor(input_dim, hidden_dim, num_layers, output_dim).to(device)

    # ✅ 손실 함수 및 옵티마이저
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ 모델 학습
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    # ✅ 모델 저장
    torch.save(model.state_dict(), "lstm_model_mac.pth")
    print("✅ 모델 저장 완료: lstm_model_mac.pth")

# 실행
if __name__ == "__main__":
    main()
