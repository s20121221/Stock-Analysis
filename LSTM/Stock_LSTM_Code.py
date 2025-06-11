import sqlite3
import sys,os
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)
from Tool.DataBaseTool import DataBaseTool
# ===== 參數設定 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, '..', 'db', 'stock.db')
MODEL_DIR = os.path.join(BASE_DIR, 'stock_Model')

STOCK_ID   = '2330.TW'
MODEL_FILENAME = f"{STOCK_ID}.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
DATA_END   = dt.date(2025, 4, 1)
DATA_START = DATA_END - relativedelta(years=2)
DBTOOL = DataBaseTool(DB_PATH)
# LSTM 參數
SEQ_LEN    = 30
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 讀 SQLite 資料 =====
rows = DBTOOL.DBSelect(
    table  = "daily_data",
    column = ["date", "open_price","high_price","low_price","close_price","volume"],
    name   = "stock_id",
    value  = STOCK_ID
)

# 檢查是否有資料
if not rows:
    raise RuntimeError("找不到符合條件的資料")

# 轉成 DataFrame，並命名欄位
df = pd.DataFrame(
    rows,
    columns=["日期原始", "開盤價", "最高價", "最低價", "收盤價", "成交量"]
)

# 解析日期欄位
df["日期"] = pd.to_datetime(df["日期原始"])
df.drop(columns="日期原始", inplace=True)

# ===== 前處理 =====
df.sort_values("日期", inplace=True)
df.reset_index(drop=True, inplace=True)

# 特徵矩陣：(N, 5)
features = df[['開盤價','最高價','最低價','收盤價','成交量']].astype(float).values
# 目標：下一步的收盤價 (scalar)
targets  = df['收盤價'].astype(float).values

# ===== Dataset & DataLoader =====
class StockDataset(Dataset):
    def __init__(self, features, targets, seq_len):
        # 分開兩個 scaler
        self.feature_scaler = MinMaxScaler()
        self.target_scaler  = MinMaxScaler()
        # fit_transform 各自處理
        self.X = self.feature_scaler.fit_transform(features)           # (N,5)
        self.y = self.target_scaler.fit_transform(targets.reshape(-1,1))  # (N,1)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]        # (seq_len, 5)
        y_next = self.y[idx + self.seq_len]            # (1,)
        # to tensors
        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_next, dtype=torch.float32)
        )

dataset = StockDataset(features, targets, SEQ_LEN)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== LSTM 模型 =====
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, seq_len, 5)
        out, _ = self.lstm(x)            # out: (B, seq_len, hidden_size)
        return self.fc(out[:, -1, :])    # 取最後 timestep

model     = LSTMModel(input_size=5).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===== 訓練迴圈 =====
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)               # (B,1)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    if epoch % 20 == 0:
        avg = total_loss / len(dataset)
        print(f"[Epoch {epoch:4d}] Loss: {avg:.6f}")

# ===== 下一日預測 =====
# 取最後 SEQ_LEN 筆特徵
last_feat   = features[-SEQ_LEN:]                           # (SEQ_LEN,5)
last_scaled = dataset.feature_scaler.transform(last_feat)   # (SEQ_LEN,5)
inp = torch.tensor(last_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

model.eval()
with torch.no_grad():
    pred_scaled = model(inp).cpu().numpy()    # shape (1,1)

next_close = dataset.target_scaler.inverse_transform(pred_scaled)[0,0]

next_date = DATA_END + dt.timedelta(days=1)
print(f"預測 {next_date} 收盤價：{next_close:.2f}")

os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"模型參數已儲存到：{MODEL_PATH}")
