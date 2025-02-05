# lstm_autoencoder_train.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time

# Device 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using device: Mac GPU (Metal MPS)")
else:
    device = torch.device("cpu")
    print("⚠️ MPS 지원되지 않음. CPU 사용")

def safe_eval(x):
    try:
        return eval(x) if isinstance(x, str) else np.nan
    except Exception as e:
        print(f"❌ safe_eval() 오류: {x} | {e}")
        return np.nan

def load_and_merge_data(folder_path):
    """
    각 CSV 파일에서 데이터를 읽어,
      - first: actual_frequencies → F1~F9, mad, entropy → mad_first, entropy_first  
      - second: actual_frequencies → S0~S9, mad, entropy → mad_second, entropy_second  
    를 생성한 후 symbol, start_date, end_date 기준 inner join하여 한 row에 23개 피처를 생성합니다.
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    data_frames = []
    
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df["start_date"] = pd.to_datetime(df["start_date"], errors='coerce')
        df["end_date"] = pd.to_datetime(df["end_date"], errors='coerce')
        if "actual_frequencies" in df.columns:
            df["actual_frequencies"] = df["actual_frequencies"].apply(safe_eval)
        
        df_first = df[df["digit_type"] == "first"].copy()
        df_second = df[df["digit_type"] == "second"].copy()
        
        if not df_first.empty:
            first_features = pd.DataFrame(
                df_first["actual_frequencies"].tolist(),
                columns=[f"F{i+1}" for i in range(9)],
                index=df_first.index
            )
            df_first = df_first.join(first_features)
            df_first = df_first.rename(columns={"mad": "mad_first", "entropy": "entropy_first"})
            cols_first = ["symbol", "start_date", "end_date", "mad_first", "entropy_first"] + [f"F{i+1}" for i in range(9)]
            df_first = df_first[cols_first]
        
        if not df_second.empty:
            second_features = pd.DataFrame(
                df_second["actual_frequencies"].tolist(),
                columns=[f"S{i}" for i in range(10)],
                index=df_second.index
            )
            df_second = df_second.join(second_features)
            df_second = df_second.rename(columns={"mad": "mad_second", "entropy": "entropy_second"})
            cols_second = ["symbol", "start_date", "end_date", "mad_second", "entropy_second"] + [f"S{i}" for i in range(10)]
            df_second = df_second[cols_second]
        
        if not df_first.empty and not df_second.empty:
            df_merged = pd.merge(df_first, df_second, on=["symbol", "start_date", "end_date"], how="inner")
            data_frames.append(df_merged)
        else:
            print(f"⚠️ {file} 파일: first 또는 second 데이터 부족")
    
    if data_frames:
        df_all = pd.concat(data_frames, ignore_index=True)
        df_all = df_all.sort_values(by=["symbol", "start_date"]).reset_index(drop=True)
        return df_all
    else:
        return pd.DataFrame()

# Autoencoder 학습용 Dataset
class CryptoTimeSeriesAutoencoderDataset(Dataset):
    def __init__(self, df, sequence_length=30):
        self.sequence_length = sequence_length
        self.X = []
        # 23개 피처: mad_first, entropy_first, mad_second, entropy_second, F1~F9, S0~S9
        feature_cols = ["mad_first", "entropy_first", "mad_second", "entropy_second"] + \
                       [f"F{i+1}" for i in range(9)] + [f"S{i}" for i in range(10)]
        for symbol, group in df.groupby("symbol"):
            group = group.sort_values(by="start_date")
            features = group[feature_cols].values
            for i in range(len(features) - sequence_length + 1):
                seq = features[i:i+sequence_length]
                self.X.append(seq)
        print(f"✅ Autoencoder 학습용 데이터셋 생성 완료! 총 샘플 수: {len(self.X)}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.X[idx], dtype=torch.float32)
        return seq, seq

def split_data_by_time(df, test_ratio=0.2):
    """
    각 symbol 별로 start_date 기준 정렬 후, 최신 test_ratio(%) 데이터를 Validation으로 분리.
    """
    train_list = []
    val_list = []
    
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values(by="start_date")  # 날짜순 정렬
        split_idx = int(len(group) * (1 - test_ratio))  # 최신 20%를 Validation으로 분리
        train_list.append(group.iloc[:split_idx])  # 과거 80% → Training
        val_list.append(group.iloc[split_idx:])  # 최신 20% → Validation

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    
    return train_df, val_df

# LSTM Autoencoder 모델
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim, latent_dim, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        enc_out, (h, c) = self.encoder_lstm(x)
        h_last = h[-1]
        latent = self.fc_enc(h_last)
        dec_h0 = self.fc_dec(latent)
        dec_h0 = dec_h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        dec_c0 = torch.zeros_like(dec_h0).to(x.device)
        dec_input = torch.zeros(batch_size, self.seq_len, dec_h0.size(-1)).to(x.device)
        dec_out, _ = self.decoder_lstm(dec_input, (dec_h0, dec_c0))
        out = self.output_layer(dec_out)
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_val_loss = 0

        # 🔹 Training (훈련)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # 🔹 Validation (검증)
        model.eval()  # 검증 모드
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

        model.train()  # 다시 훈련 모드로 변경

def main():
    train_folder = "./crypto_data/TraingData/Total_CSV/1.1_BN_Train/"
    df_train = load_and_merge_data(train_folder)

    if df_train.empty:
        print("❌ 훈련 데이터가 없습니다.")
        return

    # ✅ 최신 20% 데이터를 Validation 데이터로 분리
    train_df, val_df = split_data_by_time(df_train, test_ratio=0.2)

    print(f"✅ Train 데이터 크기: {len(train_df)}, Validation 데이터 크기: {len(val_df)}")

    # ✅ Dataset 및 DataLoader 생성
    train_dataset = CryptoTimeSeriesAutoencoderDataset(train_df, sequence_length=30)
    val_dataset = CryptoTimeSeriesAutoencoderDataset(val_df, sequence_length=30)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ✅ LSTM Autoencoder 모델 초기화
    input_dim = 23  # 23개 feature
    hidden_dim = 64
    latent_dim = 32
    num_layers = 2
    model = LSTMAutoencoder(seq_len=30, input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers).to(device)

    # ✅ 손실 함수 및 옵티마이저 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ 모델 학습 (Validation Loss 추가)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)

    # ✅ 모델 저장
    torch.save(model.state_dict(), "lstm_autoencoder.pth")
    print("✅ 모델 저장 완료: lstm_autoencoder.pth")

if __name__ == "__main__":
    main()
