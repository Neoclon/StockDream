# lstm_autoencoder_train.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

# Device 설정: GPU (Colab에서는 CUDA 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def safe_eval(x):
    try:
        return eval(x) if isinstance(x, str) else np.nan
    except Exception as e:
        print(f"Error in safe_eval: {x} | {e}")
        return np.nan

def load_data_csv(file_path):
    df = pd.read_csv(file_path)
    df["start_date"] = pd.to_datetime(df["start_date"], errors='coerce')
    df["end_date"] = pd.to_datetime(df["end_date"], errors='coerce')
    if "actual_frequencies" in df.columns:
        df["actual_frequencies"] = df["actual_frequencies"].apply(safe_eval)
    # 만약 이미 F1 컬럼이 없다면 추가 (즉, 데이터에 전처리가 되어있지 않으면)
    if "digit_type" in df.columns:
        first_mask = df["digit_type"] == "first"
        if first_mask.any() and "F1" not in df.columns:
            first_features = pd.DataFrame(
                df.loc[first_mask, "actual_frequencies"].tolist(),
                columns=[f"F{i+1}" for i in range(9)],
                index=df.loc[first_mask].index
            )
            df = df.join(first_features)
            df = df.rename(columns={"mad": "mad_first", "entropy": "entropy_first"})
        second_mask = df["digit_type"] == "second"
        if second_mask.any() and "S0" not in df.columns:
            second_features = pd.DataFrame(
                df.loc[second_mask, "actual_frequencies"].tolist(),
                columns=[f"S{i}" for i in range(10)],
                index=df.loc[second_mask].index
            )
            df = df.join(second_features)
            df = df.rename(columns={"mad": "mad_second", "entropy": "entropy_second"})
    return df

# Dataset 클래스 정의 (나머지 코드는 그대로 유지)
class LSTMAutoencoderDataset(Dataset):
    def __init__(self, df, sequence_length=15, mode="first"):
        self.sequence_length = sequence_length
        self.mode = mode
        # 만약 이미 'mad_first' 컬럼이 있으면 해당 컬럼을 사용하고,
        # 그렇지 않으면 'mad', 'entropy' 컬럼을 사용하도록 함.
        if mode == "first":
            if "mad_first" in df.columns and "entropy_first" in df.columns:
                feature_cols = ["mad_first", "entropy_first"] + [f"F{i+1}" for i in range(9)]
            else:
                feature_cols = ["mad", "entropy"] + [f"F{i+1}" for i in range(9)]
        else:
            if "mad_second" in df.columns and "entropy_second" in df.columns:
                feature_cols = ["mad_second", "entropy_second"] + [f"S{i}" for i in range(10)]
            else:
                feature_cols = ["mad", "entropy"] + [f"S{i}" for i in range(10)]
        
        self.X = []
        for symbol, group in df.groupby("symbol"):
            group = group.sort_values(by="start_date")
            features = group[feature_cols].values
            for i in range(len(features) - sequence_length + 1):
                seq = features[i:i+sequence_length]
                self.X.append(seq)
        print(f"{mode.upper()} dataset created with {len(self.X)} samples.")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.X[idx], dtype=torch.float32)
        return seq, seq

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

def main():
    # Train data CSV 경로 (예: Google Drive 경로)
    train_csv = "./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/train_data.csv"
    df_train = load_data_csv(train_csv)
    df_train = df_train[df_train["label"] == 0]  # 정상 데이터만 사용
    print("Train data loaded:", len(df_train))
    
    # First와 Second 그룹 분리
    df_train_first = df_train[df_train["digit_type"]=="first"].copy()
    df_train_second = df_train[df_train["digit_type"]=="second"].copy()
    
    sequence_length = 15
    batch_size = 32
    
    first_dataset = LSTMAutoencoderDataset(df_train_first, sequence_length, mode="first")
    second_dataset = LSTMAutoencoderDataset(df_train_second, sequence_length, mode="second")
    
    first_loader = DataLoader(first_dataset, batch_size=batch_size, shuffle=True)
    second_loader = DataLoader(second_dataset, batch_size=batch_size, shuffle=True)
    
    # Model parameters
    input_dim_first = 11   # mad_first, entropy_first + F1~F9
    input_dim_second = 12  # mad_second, entropy_second + S0~S9
    hidden_dim = 64
    latent_dim = 32
    num_layers = 2
    
    first_model = LSTMAutoencoder(seq_len=sequence_length, input_dim=input_dim_first,
                                  hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers).to(device)
    second_model = LSTMAutoencoder(seq_len=sequence_length, input_dim=input_dim_second,
                                   hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers).to(device)
    
    criterion = nn.MSELoss()
    optimizer_first = optim.Adam(first_model.parameters(), lr=0.001)
    optimizer_second = optim.Adam(second_model.parameters(), lr=0.001)
    
    print("Training First model...")
    for epoch in range(50):
        first_model.train()
        epoch_loss = 0.0
        for inputs, targets in first_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_first.zero_grad()
            outputs = first_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_first.step()
            epoch_loss += loss.item() * inputs.size(0)
        avg_loss = epoch_loss / len(first_loader.dataset)
        print(f"First Epoch {epoch+1}/50, Loss: {avg_loss:.6f}")
    
    torch.save(first_model.state_dict(), "./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/lstm_autoencoder_first.pth")
    print("First model saved.")
    
    print("Training Second model...")
    for epoch in range(50):
        second_model.train()
        epoch_loss = 0.0
        for inputs, targets in second_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_second.zero_grad()
            outputs = second_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_second.step()
            epoch_loss += loss.item() * inputs.size(0)
        avg_loss = epoch_loss / len(second_loader.dataset)
        print(f"Second Epoch {epoch+1}/50, Loss: {avg_loss:.6f}")
    
    torch.save(second_model.state_dict(), "./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/lstm_autoencoder_second.pth")
    print("Second model saved.")

if __name__ == "__main__":
    main()
