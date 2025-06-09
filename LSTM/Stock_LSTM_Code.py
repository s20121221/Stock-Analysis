import requests, pandas as pd, datetime as dt, numpy as np, torch, json
from dateutil.relativedelta import relativedelta
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===== 參數設定 =====
STOCK_ID   = '2330'
DATA_END   = dt.date(2025, 5, 6)
DATA_START = DATA_END - relativedelta(years=2)
SEQ_LEN, BATCH_SIZE, EPOCHS, LR = 30, 32, 1000, 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 抓取月資料 =====
def fetch_monthly(stock_id, date):
    date_str = date.replace(day=1).strftime("%Y%m%d")
    url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_str}&stockNo={stock_id}"
    try:
        res = requests.get(url, verify=False, timeout=10)
        res.raise_for_status()
        data = res.json()
        if data.get('data') is None:
            return pd.DataFrame()
        return pd.DataFrame(data['data'], columns=data['fields'])
    except:
        return pd.DataFrame()

dfs, curr = [], DATA_START.replace(day=1)
while curr <= DATA_END.replace(day=1):
    df = fetch_monthly(STOCK_ID, curr)
    if not df.empty:
        dfs.append(df)
    curr += relativedelta(months=1)

if not dfs:
    raise RuntimeError("無法取得資料")
df = pd.concat(dfs, ignore_index=True)

# ===== 整理資料 =====
def roc_to_ad(x):  # 民國轉西元
    y, m, d = x.replace('/', '-').split('-')
    return dt.datetime(int(y) + 1911, int(m), int(d))

df['日期'] = df['日期'].apply(roc_to_ad)
df['收盤價'] = df['收盤價'].str.replace(',', '').astype(float)
df.sort_values('日期', inplace=True)
df.reset_index(drop=True, inplace=True)

train_df = df[df['日期'].dt.date <= DATA_END]

# ===== Dataset 與 DataLoader =====
class StockDataset(Dataset):
    def __init__(self, prices, seq_len):
        self.scaler = MinMaxScaler()
        self.prices = self.scaler.fit_transform(prices.reshape(-1, 1))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.prices) - self.seq_len

    def __getitem__(self, idx):
        x = self.prices[idx:idx + self.seq_len]
        y = self.prices[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_dataset = StockDataset(train_df['收盤價'].values, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== LSTM 模型定義 =====
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===== 模型訓練 =====
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_dataset):.6f}")

# ===== 預測未來一天 =====
last_seq = train_df['收盤價'].values[-SEQ_LEN:].reshape(-1, 1)
inp = torch.tensor(train_dataset.scaler.transform(last_seq), dtype=torch.float32).unsqueeze(0).to(DEVICE)

model.eval()
with torch.no_grad():
    pred = model(inp).cpu().numpy()
pred_price = train_dataset.scaler.inverse_transform(pred)[0, 0]

next_date = DATA_END + dt.timedelta(days=1)
actual_row = df[df['日期'].dt.date == next_date]
actual_price = actual_row['收盤價'].values[0] if not actual_row.empty else float('nan')

# ===== 輸出結果 =====
print(f"\n資料最終日：{DATA_END}")
print(f"預測 {next_date} 收盤價：{pred_price:.2f}")
print(f"實際 {next_date} 收盤價：{actual_price:.2f}")
