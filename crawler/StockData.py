# 未使用此程式碼，僅供測試yfinance套件
import yfinance as yf
import datetime as dt
import pandas as pd

stock_id = "0056.TW"
end = dt.date.today()
start = end - dt.timedelta(days=365)

# 下載股票資料
stock_data = yf.download(stock_id, period="5d")
# stock_date = yf.download(stock_id, interval="1m")
stock_data = stock_data.rename(
    columns={
        "Open": "開盤價",
        "High": "最高價",
        "Low": "最低價",
        "Close": "收盤價",
        "Adj Close": "調整後收盤價",
        "Volume": "成交量",
    }
)

# 顯示設定，處理中文字對齊與浮點數格式
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.float_format", '{:.2f}'.format)

# 印出股價資料
print("\n 近5日股價資料：")
print(stock_data.to_string(justify="center"))

# 獲取公司基本資料
stock = yf.Ticker(stock_id)
info = stock.info

print("\n 公司基本資料：")
for key, value in info.items():
    print(f"{key:<30}: {value}")

# 獲取財務報表（損益表）
financials = stock.financials
print("\n 財務報表（損益表）：")
print(financials.to_string(justify="center"))
