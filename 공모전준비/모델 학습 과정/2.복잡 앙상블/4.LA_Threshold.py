# lstm_autoencoder_threshold.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def safe_eval(x):
    try:
        return eval(x) if isinstance(x, str) else np.nan
    except Exception as e:
        print(f"Error in safe_eval: {x} | {e}")
        return np.nan

def load_data_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=["start_date", "end_date"])
    if "actual_frequencies" in df.columns:
        df["actual_frequencies"] = df["actual_frequencies"].apply(safe_eval)
    if "digit_type" in df.columns:
        first_mask = df["digit_type"]=="first"
        if first_mask.any() and "F1" not in df.columns:
            first_features = pd.DataFrame(df.loc[first_mask, "actual_frequencies"].tolist(),
                                          columns=[f"F{i+1}" for i in range(9)],
                                          index=df.loc[first_mask].index)
            df = df.join(first_features)
            df = df.rename(columns={"mad": "mad_first", "entropy": "entropy_first"})
        second_mask = df["digit_type"]=="second"
        if second_mask.any() and "S0" not in df.columns:
            second_features = pd.DataFrame(df.loc[second_mask, "actual_frequencies"].tolist(),
                                           columns=[f"S{i}" for i in range(10)],
                                           index=df.loc[second_mask].index)
            df = df.join(second_features)
            df = df.rename(columns={"mad": "mad_second", "entropy": "entropy_second"})
    return df

class LSTMAutoencoderDataset(Dataset):
    def __init__(self, df, sequence_length=15, mode="first"):
        self.sequence_length = sequence_length
        self.mode = mode
        if mode=="first":
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
        self.labels = []  # ground truth label from valid set
        for symbol, group in df.groupby("symbol"):
            group = group.sort_values(by="start_date")
            data = group[feature_cols].values
            lbls = group["label"].values
            for i in range(len(data) - sequence_length + 1):
                seq = data[i:i+sequence_length]
                self.X.append(seq)
                self.labels.append(lbls[i+sequence_length-1])
        print(f"{mode.upper()} dataset created with {len(self.X)} samples.")
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        seq = torch.tensor(self.X[idx], dtype=torch.float32)
        label = self.labels[idx]
        return seq, seq, label

def custom_collate(batch):
    sequences, targets, labels = zip(*batch)
    sequences = torch.stack(sequences, 0)
    targets = torch.stack(targets, 0)
    return sequences, targets, torch.tensor(labels)

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

def evaluate_model(model, dataloader):
    model.eval()
    recon_errors = []
    true_labels = []
    with torch.no_grad():
        for inputs, targets, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = torch.mean((inputs - outputs)**2, dim=(1,2))
            recon_errors.extend(loss.cpu().numpy().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())
    return np.array(recon_errors), np.array(true_labels)

def determine_threshold(y_true, scores):
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # F1 score 계산
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    print(f"Determined threshold (max F1): {best_threshold:.6f}")
    return best_threshold

def main():
    valid_csv = "./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/valid_data.csv"
    df_valid = load_data_csv(valid_csv)

    # 분리: First와 Second 그룹
    df_valid_first = df_valid[df_valid["digit_type"]=="first"].copy()
    df_valid_second = df_valid[df_valid["digit_type"]=="second"].copy()

    sequence_length = 15
    batch_size = 32

    first_dataset = LSTMAutoencoderDataset(df_valid_first, sequence_length, mode="first")
    second_dataset = LSTMAutoencoderDataset(df_valid_second, sequence_length, mode="second")

    first_loader = DataLoader(first_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    second_loader = DataLoader(second_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    input_dim_first = 11
    input_dim_second = 12
    hidden_dim = 64
    latent_dim = 32
    num_layers = 2

    first_model = LSTMAutoencoder(seq_len=sequence_length, input_dim=input_dim_first,
                                  hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers).to(device)
    second_model = LSTMAutoencoder(seq_len=sequence_length, input_dim=input_dim_second,
                                   hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers).to(device)

    first_model.load_state_dict(torch.load("./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/lstm_autoencoder_first.pth", map_location=device))
    second_model.load_state_dict(torch.load("./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/lstm_autoencoder_second.pth", map_location=device))

    recon_first, labels_first = evaluate_model(first_model, first_loader)
    recon_second, labels_second = evaluate_model(second_model, second_loader)

    print("Validation First group: recon error stats: min={:.6f}, max={:.6f}".format(recon_first.min(), recon_first.max()))
    print("Validation Second group: recon error stats: min={:.6f}, max={:.6f}".format(recon_second.min(), recon_second.max()))

    threshold_first = determine_threshold(labels_first, recon_first)
    threshold_second = determine_threshold(labels_second, recon_second)

    thresholds = {"first": threshold_first, "second": threshold_second}
    with open("./공모전준비/모델 학습 과정/2.복잡 앙상블/결과/autoencoder_thresholds.json", "w") as f:
        import json
        json.dump(thresholds, f)
    print("Thresholds saved:", thresholds)

if __name__ == "__main__":
    main()
