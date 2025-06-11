import os
import datetime as dt
from Stock_LSTM_Code import StockPredictor

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH  = os.path.join(BASE_DIR, '..','db', 'stock.db')
    
    predictor = StockPredictor(
        db_path=DB_PATH,
        stock_id='2330',
        seq_len=30,
        batch_size=32,
        epochs=100,
        lr=1e-3,
        model_dir=os.path.join(BASE_DIR, 'stock_Model')
    )
    predictor.train()
    next_date, next_close = predictor.predict_next_day()
    print(f"預測 {next_date} 收盤價：{next_close:.2f}")