import os
import sys
import sqlite3
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from contextlib import closing
from typing import Union, List, Optional, Dict, Any,Tuple

# 將專案根目錄加入路徑
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)
from Tool.DataBaseTool import DataBaseTool

class StockDataset(Dataset):
    def __init__(self, features, targets, seq_len):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler  = MinMaxScaler()
        self.X = self.feature_scaler.fit_transform(features)
        self.y = self.target_scaler.fit_transform(targets.reshape(-1,1))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_next = self.y[idx + self.seq_len]
        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_next, dtype=torch.float32)
        )

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class StockPredictor:
    def __init__(self,
                 db_path: str,
                 stock_id: str,
                 seq_len: int = 30,
                 batch_size: int = 32,
                 epochs: int = 100,
                 lr: float = 1e-3,
                 model_dir: str = 'stock_Model',
                 device: torch.device = None):
        self.DB_PATH = db_path
        self.STOCK_ID = stock_id
        self.SEQ_LEN = seq_len
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LR = lr
        self.MODEL_DIR = model_dir
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self.MODEL_PATH = os.path.join(self.MODEL_DIR, f"{self.STOCK_ID}.pth")
        self.DEVICE = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_data()
        self._build_model()

    def _load_data(self):
        db = DataBaseTool(self.DB_PATH)
        rows = db.select(
            table='daily_data',
            columns=['date','open_price','high_price','low_price','close_price','volume'],
            where={'stock_id': self.STOCK_ID},
            order_by='date'
        )
        if not rows:
            raise RuntimeError(f"找不到 {self.STOCK_ID} 的資料")
        df = pd.DataFrame(rows)
        df.rename(columns={
            'date': '日期原始',
            'open_price': '開盤價',
            'high_price': '最高價',
            'low_price': '最低價',
            'close_price': '收盤價',
            'volume': '成交量'
        }, inplace=True)
        df['日期'] = pd.to_datetime(df['日期原始'])
        df.sort_values('日期', inplace=True)
        self.dates = [d.date() for d in df['日期']]

        features = df[['開盤價','最高價','最低價','收盤價','成交量']].astype(float).values
        targets  = df['收盤價'].astype(float).values

        self.dataset = StockDataset(features, targets, self.SEQ_LEN)
        self.loader = DataLoader(self.dataset,
                                 batch_size=self.BATCH_SIZE,
                                 shuffle=True)
        self.features = features
        self.targets = targets

    def _build_model(self):
        self.model = LSTMModel(input_size=5).to(self.DEVICE)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

    def train_one_epoch(self, verbose: bool = False) -> float:
        self.model.train()
        total_loss = 0.0
        for xb, yb in self.loader:
            xb, yb = xb.to(self.DEVICE), yb.to(self.DEVICE)
            self.optimizer.zero_grad()
            pred = self.model(xb)
            loss = self.criterion(pred, yb)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(self.dataset)
        if verbose:
            print(f"  Epoch 單次 Loss: {avg_loss:.6f}")
        return avg_loss

    def train(self, verbose: bool = True):
        for epoch in range(1, self.EPOCHS + 1):
            avg = self.train_one_epoch(verbose=verbose and (epoch % 20 == 0))
            if verbose and epoch % 20 == 0:
                print(f"[Epoch {epoch:4d}] Loss: {avg:.6f}")
        self.save_model()
        print(f"訓練結束，模型已保存至: {self.MODEL_PATH}")

    def predict_next_day(
        self,
        target_date: Union[str, dt.date, pd.Timestamp]) -> Tuple[dt.date, float]:
        # 1. 轉成 datetime.date
        if not isinstance(target_date, dt.date):
            target_date = pd.to_datetime(target_date).date()

        # 2. 確認日期存在於 self.dates 中
        try:
            idx = self.dates.index(target_date)
        except ValueError:
            raise ValueError(f"指定的日期 {target_date} 不在資料範圍內。")

        # 3. 確保有足夠的歷史資料做序列
        if idx + 1 < self.SEQ_LEN:
            raise ValueError(
                f"日期 {target_date} 前面至少要有 {self.SEQ_LEN} 天資料，"
                f"目前僅有 {idx+1} 天。"
            )

        # 4. 取出從 (idx-SEQ_LEN+1) 到 idx 的 feature
        start_idx = idx + 1 - self.SEQ_LEN
        last_feat = self.features[start_idx: idx+1]  # shape=(SEQ_LEN, n_features)

        # 5. 標準化並轉成 tensor
        last_scaled = self.dataset.feature_scaler.transform(last_feat)
        inp = torch.tensor(last_scaled, dtype=torch.float32) \
                .unsqueeze(0) \
                .to(self.DEVICE)

        # 6. 模型預測
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(inp).cpu().numpy()  # shape=(1,1) or (1,n)
        pred = self.dataset.target_scaler.inverse_transform(pred_scaled)[0, 0]

        # 7. 計算下一個交易日（簡化為 +1 天）
        next_date = target_date + dt.timedelta(days=1)
        return next_date, float(pred)

    def save_model(self, model_path: str):
        """
        :param model_path: 含目錄與檔名前綴的完整路徑，
                           例如 '../../LSTM/stock_Model'
        """
        # 1. 確保路徑有 .pth 副檔名
        if not model_path.endswith('.pth'):
            model_path += '.pth'

        # 2. 建立必要的資料夾
        folder = os.path.dirname(model_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        # 3. 儲存 state_dict
        torch.save(self.model.state_dict(), model_path)
        print(f"[Info] 模型已儲存到：{model_path}")

    def load_model(self, path: str = None):
        path = path or self.MODEL_PATH
        self.model.load_state_dict(torch.load(path, map_location=self.DEVICE))
        self.model.eval()
        print(f"模型已從 {path} 加載，並準備推斷模式。")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stock Predictor')
    parser.add_argument('--db', type=str, required=True, help='資料庫路徑')
    parser.add_argument('--stock', type=str, required=True, help='股票代碼，如 2330.TW')
    parser.add_argument('--train', action='store_true', help='執行訓練')
    args = parser.parse_args()

    predictor = StockPredictor(
        db_path=args.db,
        stock_id=args.stock
    )
    if args.train:
        predictor.train()
    else:
        predictor.load_model()

    next_date, next_price = predictor.predict_next_day()
    print(f"預測日期：{next_date}，預測收盤價：{next_price:.2f}")
