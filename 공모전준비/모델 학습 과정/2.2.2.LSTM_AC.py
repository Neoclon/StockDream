# lstm_autoencoder_train.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

def load_lstm_data_separate(folder_path):
    """
    First와 Second 데이터를 개별적으로 처리하여 독립적인 시퀀스를 생성
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    first_data_frames = []
    second_data_frames = []
    
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        
        # 날짜 변환
        df["start_date"] = pd.to_datetime(df["start_date"], errors='coerce')
        df["end_date"] = pd.to_datetime(df["end_date"], errors='coerce')
        
        # actual_frequencies 변환
        if "actual_frequencies" in df.columns:
            df["actual_frequencies"] = df["actual_frequencies"].apply(safe_eval)
        
        # 'first' 데이터 분리
        df_first = df[df["digit_type"] == "first"].copy()
        if not df_first.empty:
            first_features = pd.DataFrame(
                df_first["actual_frequencies"].tolist(),
                columns=[f"F{i+1}" for i in range(9)],
                index=df_first.index
            )
            df_first = df_first.join(first_features)
            df_first = df_first.rename(columns={"mad": "mad_first", "entropy": "entropy_first"})
            df_first = df_first[["symbol", "start_date", "end_date", "mad_first", "entropy_first"] + [f"F{i+1}" for i in range(9)]]
            first_data_frames.append(df_first)
        
        # 'second' 데이터 분리
        df_second = df[df["digit_type"] == "second"].copy()
        if not df_second.empty:
            second_features = pd.DataFrame(
                df_second["actual_frequencies"].tolist(),
                columns=[f"S{i}" for i in range(10)],
                index=df_second.index
            )
            df_second = df_second.join(second_features)
            df_second = df_second.rename(columns={"mad": "mad_second", "entropy": "entropy_second"})
            df_second = df_second[["symbol", "start_date", "end_date", "mad_second", "entropy_second"] + [f"S{i}" for i in range(10)]]
            second_data_frames.append(df_second)
    
    df_first_all = pd.concat(first_data_frames, ignore_index=True) if first_data_frames else pd.DataFrame()
    df_second_all = pd.concat(second_data_frames, ignore_index=True) if second_data_frames else pd.DataFrame()
    
    return df_first_all, df_second_all

# Autoencoder 학습용 Dataset
class CryptoTimeSeriesAutoencoderDataset(Dataset):
    def __init__(self, df, sequence_length=15, mode="first"):
        """
        df: First 또는 Second 데이터프레임
        sequence_length: 시퀀스 길이
        mode: "first" -> First Digit (F1~F9), "second" -> Second Digit (S0~S9)
        """
        self.sequence_length = sequence_length
        self.mode = mode  # first / second 선택
        
        if mode == "first":
            feature_cols = ["mad_first", "entropy_first"] + [f"F{i+1}" for i in range(9)]
        else:
            feature_cols = ["mad_second", "entropy_second"] + [f"S{i}" for i in range(10)]
        
        self.X = []
        
        for symbol, group in df.groupby("symbol"):
            group = group.sort_values(by="start_date")
            features = group[feature_cols].values  # (num_timesteps, feature_size)
            
            for i in range(len(features) - sequence_length + 1):
                seq = features[i:i+sequence_length]  # 입력 시퀀스
                self.X.append(seq)
        
        print(f"✅ {mode.upper()} 데이터셋 생성 완료! 총 샘플 수: {len(self.X)}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.X[idx], dtype=torch.float32)
        return seq, seq

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

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

def main():
    train_folder = "./crypto_data/TraingData/Total_CSV/1.1_BN_Train/"
    df_first, df_second = load_lstm_data_separate(train_folder)

    if df_first.empty or df_second.empty:
        print("❌ First 또는 Second 데이터가 없습니다.")
        return

    print(f"✅ First 데이터 로드 완료: {len(df_first)}개 샘플")
    print(f"✅ Second 데이터 로드 완료: {len(df_second)}개 샘플")

    sequence_length = 15

    # ✅ First Dataset & DataLoader
    first_dataset = CryptoTimeSeriesAutoencoderDataset(df_first, sequence_length, mode="first")
    first_loader = DataLoader(first_dataset, batch_size=32, shuffle=True)

    # ✅ Second Dataset & DataLoader
    second_dataset = CryptoTimeSeriesAutoencoderDataset(df_second, sequence_length, mode="second")
    second_loader = DataLoader(second_dataset, batch_size=32, shuffle=True)

    # ✅ 모델 정의
    input_dim_first = 11  # mad_first, entropy_first + F1~F9
    input_dim_second = 12  # mad_second, entropy_second + S0~S9
    hidden_dim = 64
    latent_dim = 32
    num_layers = 2

    first_model = LSTMAutoencoder(seq_len=sequence_length, input_dim=input_dim_first, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers).to(device)
    second_model = LSTMAutoencoder(seq_len=sequence_length, input_dim=input_dim_second, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer_first = optim.Adam(first_model.parameters(), lr=0.001)
    optimizer_second = optim.Adam(second_model.parameters(), lr=0.001)

    # ✅ First 모델 학습
    print("🚀 First 데이터 학습 시작")
    train_model(first_model, first_loader, criterion, optimizer_first, num_epochs=10)
    torch.save(first_model.state_dict(), "lstm_autoencoder_first.pth")

    # ✅ Second 모델 학습
    print("🚀 Second 데이터 학습 시작")
    train_model(second_model, second_loader, criterion, optimizer_second, num_epochs=10)
    torch.save(second_model.state_dict(), "lstm_autoencoder_second.pth")

    print("✅ 모델 저장 완료")

if __name__ == "__main__":
    main()
