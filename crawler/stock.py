# crawler/stock.py  ── 清理後版本
import sys, os, logging, colorama
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
from colorama import Fore, Style, init as colorama_init
from Tool.DataBaseTool import DataBaseTool, DB_NAME

colorama_init(autoreset=True)

class StockCrawler:
    def __init__(self, db_path: str = DB_NAME):
        self.db = DataBaseTool(db_path)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    # ---------- 抓取 yfinance ----------
    def fetch_stock_data(self, ticker: str, start_dt: datetime, end_dt: datetime):
        try:
            return yf.Ticker(ticker).history(start=start_dt, end=end_dt)
        except Exception as e:
            self.logger.error(f"Fetch {ticker} failed: {e}")
            return None

    # ---------- 更新日線 ----------
    def update_daily_data(self, ticker: str, days: int = 365):
        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=days)

        # 查詢 DB 裡此股票最後一筆日期
        with self.db.create_connection() as conn:
            cur = conn.execute(
                "SELECT MAX(date) FROM daily_data WHERE stock_id = ?",
                (ticker,))
            row = cur.fetchone()
            if row and row[0]:
                start_dt = datetime.strptime(row[0], '%Y-%m-%d')

        data = self.fetch_stock_data(ticker, start_dt, end_dt)
        if data is None or data.empty:
            self.logger.warning(f"{ticker} 沒有新資料")
            return False

        records = [
            (
                ticker,
                idx.strftime('%Y-%m-%d'),
                r['Open'], r['High'], r['Low'], r['Close'], r['Volume']
            )
            for idx, r in data.iterrows()
        ]

        query = """INSERT OR IGNORE INTO daily_data
                   (stock_id, date, open_price, high_price, low_price,
                    close_price, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)"""

        with self.db.create_connection() as conn:
            conn.executemany(query, records)
            conn.commit()

        self.logger.info(f"{ticker} 已更新 {len(records)} 筆日線資料")
        return True

    # ---------- 更新公司資訊 ----------
    def update_company_info(self, ticker: str):
        try:
            info = yf.Ticker(ticker).info
            data = {
                "symbol":         ticker,
                "name":           info.get("longName", ""),
                "industry":       info.get("industry", ""),
                "marketCap":      info.get("marketCap", 0),
                "enterpriseValue":info.get("enterpriseValue", 0)
            }
            self.db.upsert_company_info(data)
            self.logger.info(f"{ticker} 公司資訊已更新")
            return True
        except Exception as e:
            self.logger.error(f"{ticker} 公司資訊更新失敗: {e}")
            return False


# ---------------- 測試用腳本 ----------------
if __name__ == "__main__":
    crawler = StockCrawler()

    stock_list = [
        "0050.TW",  # 元大台灣50
        "0053.TW",  # 元大電子
        "00962.TW"  # 台股AI優息動能
    ]

    for code in stock_list:
        if crawler.update_company_info(code):
            print(Fore.GREEN + f"{code} 公司資訊已更新")
        else:
            print(Fore.RED + f"{code} 公司資訊更新失敗")

        if crawler.update_daily_data(code, days=365):
            print(Fore.GREEN + f"{code} 日線資料已更新")
        else:
            print(Fore.YELLOW + f"{code} 沒有新日線資料")
