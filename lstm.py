
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로딩
df = pd.read_csv('metocean_data.csv', dtype={'TM': str})
df['TM'] = df['TM'].astype(str).str.strip().str.replace("'", "")
df['TM'] = pd.to_datetime(df['TM'], format='%Y%m%d%H%M')
df.set_index('TM', inplace=True)

# 2. 전처리
cols = ['WH', 'WD', 'WS', 'GST', 'TW', 'TA', 'PA', 'HM']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
df = df.interpolate(method='time').dropna()

# 3. 정규화
input_cols = ['WS', 'WH', 'TA', 'PA', 'HM', 'GST']
target_cols = ['WS', 'WH']

scaler_input = StandardScaler()
scaler_target = StandardScaler()

scaled_input = scaler_input.fit_transform(df[input_cols])
scaled_target = scaler_target.fit_transform(df[target_cols])

# 4. 슬라이딩 윈도우
input_len = 168
horizon = 72
X, y = [], []

for i in range(len(df) - input_len - horizon):
    X.append(scaled_input[i:i + input_len])
    y.append(scaled_target[i + input_len + horizon - 1])  # 단일 포인트 예측

X = np.array(X)
y = np.array(y)

# 5. 데이터 분할
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

# 6. 텐서 생성
train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.float32)), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                      torch.tensor(y_val, dtype=torch.float32)), batch_size=32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 7. 모델 정의
class DeepLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=X.shape[2], hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(self.dropout(x)))
        return self.fc2(x)

model = DeepLSTM()
criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 8. 학습 + EarlyStopping
EPOCHS = 300
early_stop_patience = 10
best_loss = float('inf')
early_stop_counter = 0

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            val_loss = criterion(model(xb), yb)
            val_losses.append(val_loss.item())
    avg_val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        early_stop_counter = 0
        best_model_state = model.state_dict()
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# 9. 예측 및 평가
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

y_pred_inv = scaler_target.inverse_transform(y_pred)
y_test_inv = scaler_target.inverse_transform(y_test)

true_ws, pred_ws = y_test_inv[:, 0], y_pred_inv[:, 0]
true_wh, pred_wh = y_test_inv[:, 1], y_pred_inv[:, 1]

ws_error_pct = np.abs(true_ws - pred_ws) / (true_ws + 1e-6) * 100
wh_error_pct = np.abs(true_wh - pred_wh) / (true_wh + 1e-6) * 100

print("\n====== Wind Speed Prediction Evaluation ======")
print(f"MAE  : {mean_absolute_error(true_ws, pred_ws):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(true_ws, pred_ws)):.4f}")
print(f"R²   : {r2_score(true_ws, pred_ws):.4f}")
print(f"Error(WS)% - Max: {np.max(ws_error_pct):.2f}%, Min: {np.min(ws_error_pct):.2f}%, Mean: {np.mean(ws_error_pct):.2f}%")

print("\n====== Wave Height Prediction Evaluation ======")
print(f"MAE  : {mean_absolute_error(true_wh, pred_wh):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(true_wh, pred_wh)):.4f}")
print(f"R²   : {r2_score(true_wh, pred_wh):.4f}")
print(f"Error(WH)% - Max: {np.max(wh_error_pct):.2f}%, Min: {np.min(wh_error_pct):.2f}%, Mean: {np.mean(wh_error_pct):.2f}%")

# 10. 시각화 (마지막 100개 예측)
start_idx = -100 if len(true_ws) >= 100 else 0
x_range = np.arange(1, len(true_ws[start_idx:]) + 1)

plt.figure(figsize=(10, 4))
plt.plot(x_range, true_ws[start_idx:], label='True Wind Speed')
plt.plot(x_range, pred_ws[start_idx:], label='Predicted Wind Speed', linestyle='--')
plt.title('Wind Speed Prediction (Last 100 Forecasts)')
plt.xlabel('Sample Index (each = 72-hr forecast point)')
plt.ylabel('Wind Speed (m/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(10, 4))
plt.plot(x_range, true_wh[start_idx:], label='True Wave Height')
plt.plot(x_range, pred_wh[start_idx:], label='Predicted Wave Height', linestyle='--')
plt.title('Wave Height Prediction (Last 100 Forecasts)')
plt.xlabel('Sample Index (each = 72-hr forecast point)')
plt.ylabel('Wave Height (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()