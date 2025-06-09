import sqlite3
from sqlite3 import Error
import logging
import os

class DataBaseTool:

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'Stock.db'))
        self.db_path = db_path
        self.conn = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    # 建立資料庫連接
    def create_connection(self):
        """
        Returns:
            sqlite3.Connection: 資料庫連接對象，失敗時返回 None
        """
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")  # 啟用外鍵約束
            self.logger.info("Database connection established")
            return self.conn
        except Error as e:
            self.logger.error(f"Connection error: {e}")
            return None

    # 關閉資料庫連接
    def close_connection(self):
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")

    # 執行 SQL 查詢
    def execute_query(self, query, params=None, many=False):
        """
        Args:
            query (str): SQL 查詢語句
            params (tuple/list, optional): 查詢參數
            many (bool): 是否為批量操作（使用 executemany)

        Returns:
            sqlite3.Cursor: 執行後的游標對象，失敗時返回 None
        """
        cursor = self.conn.cursor()
        try:
            if many and params:
                cursor.executemany(query, params)
            elif params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.conn.commit()
            return cursor
        except Error as e:
            self.logger.error(f"Query execution failed: {e}\nQuery: {query}")
            self.conn.rollback()
            return None

    # 獲取查詢結果
    def fetch_data(self, query, params=None):
        """
        Args:
            query (str): SQL 查詢語句
            params (tuple/list, optional): 查詢參數

        Returns:
            dict: {'columns': [欄位名稱], 'data': [查詢結果]}，失敗時返回 None
        """
        cursor = self.execute_query(query, params)
        if cursor:
            columns = [desc[0] for desc in cursor.description]
            return {"columns": columns, "data": cursor.fetchall()}
        return None

    # 初始化資料表結構
    def initialize_database(self):
        schema = [
            """CREATE TABLE IF NOT EXISTS company_info (
                stock_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                industry TEXT,
                capital INTEGER,
                market_value INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS daily_data (
                sno INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_id TEXT NOT NULL,
                date DATE NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                PB REAL,
                buy_and_sell TEXT,
                margin_trading INTEGER,
                short_selling INTEGER,
                FOREIGN KEY (stock_id) REFERENCES company_info(stock_id)
                    ON DELETE CASCADE,
                CONSTRAINT unique_daily_record UNIQUE (stock_id, date)
            )"""
        ]

        self.create_connection()
        for table_query in schema:
            self.execute_query(table_query)
        self.close_connection()
        self.logger.info("Database tables initialized")

    # 插入公司資訊
    def upsert_company_info(self, stock_id, name, industry, capital, mValue):
        """
        Args:
            stock_id (str): 股票代號
            name (str): 公司名稱
            industry (str): 產業類別
            capital (int): 資本額
            mValue (int): 市值
        """
        query = """INSERT OR REPLACE INTO company_info
                   (stock_id, name, industry, capital, mValue)
                   VALUES (?, ?, ?, ?, ?)"""
        self.execute_query(
            query,
            (stock_id, name, industry, capital, mValue)
        )
