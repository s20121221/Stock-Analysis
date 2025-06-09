import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yfinance as yf
from datetime import datetime, timedelta
from Tool.DataBaseTool import DataBaseTool
import logging
import colorama
from colorama import init, Fore, Style
init(autoreset=True)


class StockCrawler:
    def __init__(self):
        self.db = DataBaseTool()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    # 使用yfiance抓數據
    def fetch_stock_data(self, ticker, start_date, end_date):
        try:
            stock = yf.Ticker(ticker)
            return stock.history(start=start_date, end=end_date)
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {ticker}: {e}")
            return None

    # 更新股票日線數據
    def update_daily_data(self, ticker, days=365):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # 檢查資料庫最新日期
        self.db.create_connection()
        latest_date = self.db.fetch_data(
            "SELECT MAX(date) FROM daily_data WHERE stock_id = ?",
            (ticker,)
        )

        if latest_date and latest_date['data'][0][0]:
            start_date = datetime.strptime(
                latest_date['data'][0][0], '%Y-%m-%d'
            )

        # 抓yfinance資料
        data = self.fetch_stock_data(ticker, start_date, end_date)
        if data is None or data.empty:
            self.logger.warning(f"{ticker} 沒有新資料")
            return False

        # 存入資料庫
        query = """INSERT OR IGNORE INTO daily_data
                   (stock_id, date, open_price, high_price, low_price,
                    close_price, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)"""
        records = [
            (
                ticker,
                date.strftime('%Y-%m-%d'),
                row['Open'],                # 開盤價
                row['High'],                # 最高價
                row['Low'],                 # 最低價
                row['Close'],               # 收盤價
                row['Volume']               # 成交量
            )
            for date, row in data.iterrows()
        ]
        self.db.execute_query(query, records, many=True)
        self.db.close_connection()
        self.logger.info(f"已更新 {len(records)}  筆日線資料 {ticker}")
        return True

    # 更新公司資訊
    def update_company_info(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            data = (
                ticker,
                info.get('longName', ''),
                info.get('industry', ''),
                info.get('marketCap', 0),
                info.get('enterpriseValue', 0)
            )

            self.db.create_connection()
            self.db.upsert_company_info(*data)
            self.db.close_connection()
            self.logger.info(f"{ticker} 公司資訊已更新")
            return True
        except Exception as e:
            self.logger.error(f"公司資訊更新失敗: {e}")
            return False
# 抓取股票
if __name__ == "__main__":
    '''
    單獨抓一支
    crawler = StockCrawler()
    # 插入公司資料
    crawler.update_company_info("2330.TW")
    # 插入日線資料
    crawler.update_daily_data("2330.TW", )
    '''
    crawler = StockCrawler()
    stock_list = [
        "2330.TW",  # 台積電
        "2317.TW",  # 鴻海
        "2454.TW",  # 聯發科
        "0056.TW",  # 元大高股息
    ]
for stock in stock_list:
    if crawler.update_company_info(stock):
        print(Fore.GREEN + f"{stock} 公司資訊已更新")
    else:
        print(Fore.RED + f"{stock} 公司資訊更新失敗")

    if crawler.update_daily_data(stock, days=365):
        print(Fore.GREEN + f"{stock} 日線資料已更新")
    else:
        print(Fore.RED +f"{stock} 沒有新日線資料")
