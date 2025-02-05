import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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

# ✅ 데이터 로드 및 병합 함수 (LSTM용)
def load_lstm_data(folder_path):
    """
    각 CSV 파일에서 데이터를 읽은 후, 
    'first'와 'second' 행을 각각 처리하여 MAD 값을 별도의 컬럼(mad_first, mad_second)으로 만들고,
    actual_frequencies를 각각 F1~F9 (first)와 S0~S9 (second)로 변환한 뒤,
    symbol, start_date, end_date 기준으로 병합하여 하나의 row에 21개 피처(2 MAD + 9 F + 10 S)를 만듭니다.
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    data_frames = []
    
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        
        # 날짜 변환: format 인자 없이 자동으로 파싱
        df["start_date"] = pd.to_datetime(df["start_date"], errors='coerce')
        df["end_date"] = pd.to_datetime(df["end_date"], errors='coerce')
        
        # actual_frequencies 변환
        if "actual_frequencies" in df.columns:
            df["actual_frequencies"] = df["actual_frequencies"].apply(safe_eval)
        
        # 'first'와 'second' 데이터 분리
        df_first = df[df["digit_type"] == "first"].copy()
        df_second = df[df["digit_type"] == "second"].copy()
        
        # First digit 처리: actual_frequencies → F1~F9, mad → mad_first
        if not df_first.empty:
            first_features = pd.DataFrame(
                df_first["actual_frequencies"].tolist(),
                columns=[f"F{i+1}" for i in range(9)],
                index=df_first.index
            )
            df_first = df_first.join(first_features)
            df_first = df_first.rename(columns={"mad": "mad_first"})
            cols_first = ["symbol", "start_date", "end_date", "mad_first"] + [f"F{i+1}" for i in range(9)]
            df_first = df_first[cols_first]
        
        # Second digit 처리: actual_frequencies → S0~S9, mad → mad_second
        if not df_second.empty:
            second_features = pd.DataFrame(
                df_second["actual_frequencies"].tolist(),
                columns=[f"S{i}" for i in range(10)],
                index=df_second.index
            )
            df_second = df_second.join(second_features)
            df_second = df_second.rename(columns={"mad": "mad_second"})
            cols_second = ["symbol", "start_date", "end_date", "mad_second"] + [f"S{i}" for i in range(10)]
            df_second = df_second[cols_second]
        
        # symbol, start_date, end_date 기준으로 first와 second 병합 (양쪽 모두 있어야 함)
        if not df_first.empty and not df_second.empty:
            df_merged = pd.merge(df_first, df_second, on=["symbol", "start_date", "end_date"], how="inner")
            data_frames.append(df_merged)
        else:
            print(f"⚠️ 파일 {file} 에서 first 또는 second 데이터가 부족합니다.")
    
    if data_frames:
        df_all = pd.concat(data_frames, ignore_index=True)
        df_all = df_all.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
        return df_all
    else:
        return pd.DataFrame()

# ✅ LSTM 데이터셋 클래스
class CryptoTimeSeriesDataset(Dataset):
    def __init__(self, df, sequence_lengths=[15, 20, 25, 30]):
        """
        df: load_lstm_data()로 생성된 DataFrame
        sequence_lengths: 다양한 시퀀스 길이
        각 symbol별로 정렬된 데이터를 기반으로, 각 시퀀스 길이별 시계열 데이터를 생성합니다.
        Feature 컬럼: mad_first, mad_second, F1~F9, S0~S9 (총 21개)
        """
        self.sequence_lengths = sequence_lengths
        self.grouped_data = {symbol: df_group for symbol, df_group in df.groupby("symbol")}
        self.X, self.y = [], []
        
        feature_cols = ["mad_first", "mad_second"] + [f"F{i+1}" for i in range(9)] + [f"S{i}" for i in range(10)]
        
        for symbol, group in self.grouped_data.items():
            group = group.sort_values(by="start_date")
            features = group[feature_cols].values  # (num_timesteps, 21)
            num_features = features.shape[1]
            
            for seq_length in self.sequence_lengths:
                for i in range(len(features) - seq_length):
                    seq = features[i:i + seq_length]
                    if seq.shape == (seq_length, num_features):
                        self.X.append(seq)
                        # 예측 대상으로는 다음 시점의 마지막 피처(S9)를 예로 사용 (필요에 따라 변경 가능)
                        self.y.append(features[i + seq_length][-1])
        
        print(f"✅ 데이터셋 생성 완료! 총 샘플 수: {len(self.X)}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# ✅ Custom collate function: 시퀀스 길이가 서로 다를 경우 padding 처리
def custom_collate(batch):
    sequences = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])
    sequences_padded = pad_sequence(sequences, batch_first=True)
    return sequences_padded, targets

# ✅ LSTM 모델 정의
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # 마지막 시퀀스 타임스텝의 출력
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
    if df.empty:
        print("❌ 데이터가 없습니다.")
        return
    print("✅ 데이터 로드 및 병합 완료. 샘플 수:", len(df))
    
    sequence_lengths = [15, 20, 25, 30]
    dataset = CryptoTimeSeriesDataset(df, sequence_lengths)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    print(f"✅ 총 {len(dataset)}개 시퀀스 데이터 생성 완료!")
    
    # 입력 피처 수: 21 (mad_first, mad_second, F1~F9, S0~S9)
    input_dim = 21
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    
    model = LSTMPredictor(input_dim, hidden_dim, num_layers, output_dim).to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)
    
    torch.save(model.state_dict(), "lstm_model_mac.pth")
    print("✅ 모델 저장 완료: lstm_model_mac.pth")

if __name__ == "__main__":
    main()
