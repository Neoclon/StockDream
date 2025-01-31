import torch
import torch.nn as nn
import numpy as np

# Autoencoder 모델 정의
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

# Autoencoder 학습 함수
def train_anomaly_model(mac_sequences, window_size=5, epochs=50, learning_rate=0.001):
    """
    MAC 값 데이터를 사용하여 Autoencoder 학습
    :param mac_sequences: 슬라이딩 윈도우로 구성된 numpy array 데이터
    :param window_size: 슬라이딩 윈도우 크기
    :param epochs: 학습 반복 횟수
    :param learning_rate: 학습률
    :return: 학습된 Autoencoder 모델
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MACAnomalyDetector(input_dim=window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    # 학습 데이터 준비
    inputs = torch.FloatTensor(mac_sequences).to(device)
    targets = inputs.clone()

    # 모델 학습
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")

    return model

# 모델 저장 함수
def save_model(model, filepath):
    """
    학습된 모델 저장
    :param model: 학습된 Autoencoder 모델
    :param filepath: 모델 저장 경로
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

# 실행
if __name__ == "__main__":
    # 사용자 설정
    first_training_data_file = "first_training_data.npy"
    second_training_data_file = "second_training_data.npy"
    window_size = 5
    epochs = 50
    learning_rate = 0.001

    # First Digit 데이터 학습
    print("First Digit 데이터를 불러와 학습을 시작합니다...")
    first_training_data = np.load(first_training_data_file)
    first_digit_model = train_anomaly_model(
        mac_sequences=first_training_data,
        window_size=window_size,
        epochs=epochs,
        learning_rate=learning_rate
    )
    save_model(first_digit_model, filepath="mac_anomaly_model_first_digit.pth")

    # Second Digit 데이터 학습
    print("Second Digit 데이터를 불러와 학습을 시작합니다...")
    second_training_data = np.load(second_training_data_file)
    second_digit_model = train_anomaly_model(
        mac_sequences=second_training_data,
        window_size=window_size,
        epochs=epochs,
        learning_rate=learning_rate
    )
    save_model(second_digit_model, filepath="mac_anomaly_model_second_digit.pth")

    print("모든 모델 학습이 완료되었습니다!")
