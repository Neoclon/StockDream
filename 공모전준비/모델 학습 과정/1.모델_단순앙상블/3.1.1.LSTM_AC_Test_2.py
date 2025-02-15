import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Using device: CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️ CUDA GPU를 사용할 수 없습니다. CPU 사용")

def safe_eval(x):
    try:
        return eval(x) if isinstance(x, str) else np.nan
    except Exception as e:
        print(f"Error in safe_eval: {x} | {e}")
        return np.nan

def load_lstm_data_separate(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    first_data_frames = []
    second_data_frames = []
    
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df["start_date"] = pd.to_datetime(df["start_date"], errors='coerce')
        df["end_date"] = pd.to_datetime(df["end_date"], errors='coerce')
        if "actual_frequencies" in df.columns:
            df["actual_frequencies"] = df["actual_frequencies"].apply(safe_eval)
        
        # First 데이터 처리
        df_first = df[df["digit_type"]=="first"].copy()
        if not df_first.empty:
            first_features = pd.DataFrame(
                df_first["actual_frequencies"].tolist(),
                columns=[f"F{i+1}" for i in range(9)],
                index=df_first.index
            )
            df_first = df_first.join(first_features)
            df_first = df_first.rename(columns={"mad": "mad_first", "entropy": "entropy_first"})
            df_first = df_first[["symbol", "start_date", "end_date", "mad_first", "entropy_first"] 
                                + [f"F{i+1}" for i in range(9)]]
            first_data_frames.append(df_first)
        
        # Second 데이터 처리
        df_second = df[df["digit_type"]=="second"].copy()
        if not df_second.empty:
            second_features = pd.DataFrame(
                df_second["actual_frequencies"].tolist(),
                columns=[f"S{i}" for i in range(10)],
                index=df_second.index
            )
            df_second = df_second.join(second_features)
            df_second = df_second.rename(columns={"mad": "mad_second", "entropy": "entropy_second"})
            df_second = df_second[["symbol", "start_date", "end_date", "mad_second", "entropy_second"]
                                  + [f"S{i}" for i in range(10)]]
            second_data_frames.append(df_second)
    
    df_first_all = pd.concat(first_data_frames, ignore_index=True) if first_data_frames else pd.DataFrame()
    df_second_all = pd.concat(second_data_frames, ignore_index=True) if second_data_frames else pd.DataFrame()
    
    return df_first_all, df_second_all

class CryptoTimeSeriesAutoencoderDataset(Dataset):
    def __init__(self, df, sequence_length=15, mode="first"):
        self.sequence_length = sequence_length
        self.mode = mode
        
        if mode == "first":
            feature_cols = ["mad_first", "entropy_first"] + [f"F{i+1}" for i in range(9)]
        else:
            feature_cols = ["mad_second", "entropy_second"] + [f"S{i}" for i in range(10)]
        
        self.X = []
        self.meta = []  # (symbol, start_date, digit_type)
        for symbol, group in df.groupby("symbol"):
            group = group.sort_values(by="start_date")
            features = group[feature_cols].values
            meta_info = group[["symbol", "start_date"]].values
            for i in range(len(features) - sequence_length + 1):
                seq = features[i:i+sequence_length]
                self.X.append(seq)
                # meta: (symbol, 해당 시퀀스 마지막 날짜, digit_type)
                self.meta.append((symbol, meta_info[i+sequence_length-1][1], mode))
        print(f"Dataset for {mode.upper()} created with {len(self.X)} samples.")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.X[idx], dtype=torch.float32)
        return seq, seq, self.meta[idx]

def custom_collate(batch):
    sequences, targets, meta = zip(*batch)
    sequences = torch.stack(sequences, dim=0)
    targets = torch.stack(targets, dim=0)
    return sequences, targets, list(meta)

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

def test_model(model, dataloader):
    model.eval()
    results = []
    all_errors = []
    with torch.no_grad():
        for inputs, targets, meta in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # 평균 재구성 MSE per sample (시퀀스 단위)
            loss = torch.mean((inputs - outputs)**2, dim=(1,2))
            loss_np = loss.cpu().numpy()
            all_errors.extend(loss_np.tolist())
            for m, l in zip(meta, loss_np):
                results.append({"symbol": m[0], "start_date": m[1], "digit_type": m[2], "recon_error": l})
    return results, np.array(all_errors)

def main():
    # 테스트 데이터 폴더 
    test_folder = "./crypto_data/TraingData/Total_CSV/1.BN_24/2.후반기/"
    sequence_length = 15
    
    df_first, df_second = load_lstm_data_separate(test_folder)
    if df_first.empty and df_second.empty:
        print("No test data available.")
        return
    
    # Create datasets & dataloaders for first and second
    first_dataset = CryptoTimeSeriesAutoencoderDataset(df_first, sequence_length, mode="first")
    second_dataset = CryptoTimeSeriesAutoencoderDataset(df_second, sequence_length, mode="second")
    
    first_loader = DataLoader(first_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    second_loader = DataLoader(second_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    
    # Define model parameters
    input_dim_first = 11   # mad_first, entropy_first + F1~F9
    input_dim_second = 12  # mad_second, entropy_second + S0~S9
    hidden_dim = 64
    latent_dim = 32
    num_layers = 2
    
    # Load pre-trained models
    first_model = LSTMAutoencoder(seq_len=sequence_length, input_dim=input_dim_first,
                                  hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers).to(device)
    second_model = LSTMAutoencoder(seq_len=sequence_length, input_dim=input_dim_second,
                                   hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers).to(device)
    
    first_model.load_state_dict(torch.load(
        "./공모전준비/모델 학습 과정/학습 모델/YB_data/lstm_autoencoder_first.pth", 
        map_location=device))
    second_model.load_state_dict(torch.load(
        "./공모전준비/모델 학습 과정/학습 모델/YB_data/lstm_autoencoder_second.pth", 
        map_location=device))
    
    # Test first model
    print("Testing First model...")
    results_first, errors_first = test_model(first_model, first_loader)
    min_err_first = errors_first.min()
    max_err_first = errors_first.max()
    # ae_prob: (recon_error - min) / (max - min)  (0 ~ 1 스케일)
    for r in results_first:
        r["ae_prob"] = (r["recon_error"] - min_err_first) / (max_err_first - min_err_first + 1e-8)
    
    # Test second model
    print("Testing Second model...")
    results_second, errors_second = test_model(second_model, second_loader)
    min_err_second = errors_second.min()
    max_err_second = errors_second.max()
    for r in results_second:
        r["ae_prob"] = (r["recon_error"] - min_err_second) / (max_err_second - min_err_second + 1e-8)
    
    # Combine results
    all_results = results_first + results_second
    df_results = pd.DataFrame(all_results)
    
    # 결과 저장
    df_results.to_csv("./공모전준비/모델 학습 과정/결과/autoencoder_results.csv", index=False)
    print("Autoencoder test results saved to autoencoder_results.csv")

    # -----------------------------
    # First/Second 분리
    # -----------------------------
    df_first_res = df_results[df_results["digit_type"]=="first"].copy()
    df_second_res = df_results[df_results["digit_type"]=="second"].copy()

    # mean, std 계산
    mean_first  = df_first_res["ae_prob"].mean()
    std_first   = df_first_res["ae_prob"].std(ddof=1)
    mean_second = df_second_res["ae_prob"].mean()
    std_second  = df_second_res["ae_prob"].std(ddof=1)

    # 95%, 99% 임계값
    threshold_95_first = mean_first + 1.96 * std_first
    threshold_99_first = mean_first + 2.58 * std_first
    threshold_95_second = mean_second + 1.96 * std_second
    threshold_99_second = mean_second + 2.58 * std_second

    # 임계값 초과 비율(%)
    anomaly_ratio_95_first = (df_first_res["ae_prob"] > threshold_95_first).mean() * 100
    anomaly_ratio_99_first = (df_first_res["ae_prob"] > threshold_99_first).mean() * 100

    anomaly_ratio_95_second = (df_second_res["ae_prob"] > threshold_95_second).mean() * 100
    anomaly_ratio_99_second = (df_second_res["ae_prob"] > threshold_99_second).mean() * 100

    # -----------------------------
    # "한 번 이상" 초과 심볼 수
    # -----------------------------
    # First
    over_95_first_symbols = (df_first_res.groupby("symbol")["ae_prob"].max() > threshold_95_first).sum()
    over_99_first_symbols = (df_first_res.groupby("symbol")["ae_prob"].max() > threshold_99_first).sum()
    # Second
    over_95_second_symbols = (df_second_res.groupby("symbol")["ae_prob"].max() > threshold_95_second).sum()
    over_99_second_symbols = (df_second_res.groupby("symbol")["ae_prob"].max() > threshold_99_second).sum()

    # -----------------------------
    # "N번 이상" 초과 심볼 수 (예: 2번 이상)
    # -----------------------------
    N_TIMES = 5  # 원하는 횟수 설정
    # First
    exceed_counts_95_first = df_first_res.groupby("symbol")["ae_prob"].apply(lambda x: (x > threshold_95_first).sum())
    exceed_counts_99_first = df_first_res.groupby("symbol")["ae_prob"].apply(lambda x: (x > threshold_99_first).sum())

    over_95_first_symbols_n_times = (exceed_counts_95_first >= N_TIMES).sum()
    over_99_first_symbols_n_times = (exceed_counts_99_first >= N_TIMES).sum()

    # Second
    exceed_counts_95_second = df_second_res.groupby("symbol")["ae_prob"].apply(lambda x: (x > threshold_95_second).sum())
    exceed_counts_99_second = df_second_res.groupby("symbol")["ae_prob"].apply(lambda x: (x > threshold_99_second).sum())

    over_95_second_symbols_n_times = (exceed_counts_95_second >= N_TIMES).sum()
    over_99_second_symbols_n_times = (exceed_counts_99_second >= N_TIMES).sum()

    # -----------------------------
    # 결과 출력
    # -----------------------------
    print("\n=== First ===")
    print(f"Mean(ae_prob) = {mean_first:.4f}, Std(ae_prob) = {std_first:.4f}")
    print(f"95% 임계값 = {threshold_95_first:.4f}, 99% 임계값 = {threshold_99_first:.4f}")
    print(f"95% 임계값 초과 비율: {anomaly_ratio_95_first:.2f}%")
    print(f"99% 임계값 초과 비율: {anomaly_ratio_99_first:.2f}%")
    print(f"95% 임계값 초과 심볼 수(한 번 이상): {over_95_first_symbols}개")
    print(f"99% 임계값 초과 심볼 수(한 번 이상): {over_99_first_symbols}개")
    print(f"95% 임계값 {N_TIMES}번 이상 초과 심볼 수: {over_95_first_symbols_n_times}개")
    print(f"99% 임계값 {N_TIMES}번 이상 초과 심볼 수: {over_99_first_symbols_n_times}개")

    print("\n=== Second ===")
    print(f"Mean(ae_prob) = {mean_second:.4f}, Std(ae_prob) = {std_second:.4f}")
    print(f"95% 임계값 = {threshold_95_second:.4f}, 99% 임계값 = {threshold_99_second:.4f}")
    print(f"95% 임계값 초과 비율: {anomaly_ratio_95_second:.2f}%")
    print(f"99% 임계값 초과 비율: {anomaly_ratio_99_second:.2f}%")
    print(f"95% 임계값 초과 심볼 수(한 번 이상): {over_95_second_symbols}개")
    print(f"99% 임계값 초과 심볼 수(한 번 이상): {over_99_second_symbols}개")
    print(f"95% 임계값 {N_TIMES}번 이상 초과 심볼 수: {over_95_second_symbols_n_times}개")
    print(f"99% 임계값 {N_TIMES}번 이상 초과 심볼 수: {over_99_second_symbols_n_times}개")

if __name__ == "__main__":
    main()

