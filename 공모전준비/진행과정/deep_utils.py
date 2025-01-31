# deep_utils.py
import torch
import torch.nn as nn
import numpy as np

class MACAnomalyDetector(nn.Module):
    """MAC 값 시퀀스 이상 탐지용 Autoencoder"""
    def __init__(self, input_dim=1, encoding_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

def detect_anomalies(mac_sequence, window_size=5, threshold=0.015):
    """MAC 시퀀스에서 이상치 탐지"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 초기화
    model = MACAnomalyDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 데이터 준비
    sequences = []
    for i in range(len(mac_sequence) - window_size):
        seq = mac_sequence[i:i+window_size]
        sequences.append(seq)
    
    # 학습 루프
    model.train()
    for epoch in range(50):
        total_loss = 0
        for seq in sequences:
            inputs = torch.FloatTensor(seq[:-1]).unsqueeze(1).to(device)
            targets = torch.FloatTensor(seq[1:]).unsqueeze(1).to(device)
            
            outputs = model(inputs)
            loss = torch.nn.MSELoss()(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
    # 이상치 탐지
    model.eval()
    anomalies = []
    with torch.no_grad():
        for i in range(len(mac_sequence) - window_size):
            seq = mac_sequence[i:i+window_size]
            inputs = torch.FloatTensor(seq[:-1]).unsqueeze(1).to(device)
            targets = torch.FloatTensor(seq[1:]).unsqueeze(1).to(device)
            
            outputs = model(inputs)
            error = torch.mean((outputs - targets)**2).item()
            if error > threshold:
                anomalies.append(i + window_size)  # 이상치 위치 기록
    
    return anomalies