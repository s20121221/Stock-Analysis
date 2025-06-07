import yfinance as yf
import datetime as dt
import pandas as pd

stock_id = "0056.TW"
end = dt.date.today()
start = end - dt.timedelta(days=365)

# 下載股票資料
stock_date = yf.download(stock_id, period="5d")
# stock_date = yf.download(stock_id, interval="1m")
stock_date = stock_date.rename(
    columns={
        "Open": "開盤價",
        "High": "最高價",
        "Low": "最低價",
        "Close": "收盤價",
        "Adj Close": "調整後收盤價",
        "Volume": "成交量",
    }
)

stock_date.index.name = "日期"

pd.set_option("display.unicode.east_asian_width", True)
print(stock_date.to_string(justify="center", float_format="%.2f".__mod__))
