import sqlite3
from typing import List, Union, Any, Optional, Tuple, Dict
from pathlib import Path
from contextlib import closing

DB_NAME = "stock.db"
class DataBaseTool:

    def __init__(self, db_path: Union[str, Path] = "stock.db") -> None:
        """
        初始化資料庫工具
        """
        self.db_path = str(db_path)
        self._initialize_db()

    def _get_conn(self) -> sqlite3.Connection:
        """建立資料庫連接並啟用外鍵約束"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize_db(self) -> None:
        """初始化資料庫表格結構"""
        with closing(self._get_conn()) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS company_info (
                    stock_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    industry TEXT,
                    capital INTEGER,
                    mValue INTEGER
                );

                CREATE TABLE IF NOT EXISTS daily_data (
                    stock_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    PRIMARY KEY (stock_id, date),
                    FOREIGN KEY (stock_id) REFERENCES company_info(stock_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_daily_stock ON daily_data(stock_id);
                CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_data(date);
            ''')
            conn.commit()

    def select(
        self,
        table: str,
        columns: Union[str, List[str]] = "*",
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        執行 SELECT 查詢操作

        Args:
            table: 資料表名稱
            columns: 要查詢的欄位 (預設全部)
            where: 條件字典 {欄位名: 值}
            order_by: 排序條件 (e.g. "date DESC")
            limit: 限制返回筆數

        Returns:
            包含查詢結果的字典列表
        """
        try:
            # 構建查詢語句
            column_str = ", ".join(columns) if isinstance(columns, list) else columns
            query = f"SELECT {column_str} FROM {table}"
            params = []

            # 添加 WHERE 條件
            if where:
                conditions = [f"{k} = ?" for k in where.keys()]
                query += " WHERE " + " AND ".join(conditions)
                params.extend(where.values())

            # 添加排序
            if order_by:
                query += f" ORDER BY {order_by}"

            # 添加筆數限制
            if limit:
                query += f" LIMIT {limit}"

            # 執行查詢
            with closing(self._get_conn()) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            print(f"Select error: {e}")
            return []

    def insert(
        self,
        table: str,
        data: Dict[str, Any],
        on_conflict: str = "IGNORE"
    ) -> bool:
        """
        執行 INSERT 操作

        Args:
            table: 資料表名稱
            data: 要插入的資料 {欄位名: 值}
            on_conflict: 衝突處理方式 (IGNORE|REPLACE|ABORT等)

        Returns:
            是否成功
        """
        try:
            columns = list(data.keys())
            placeholders = ", ".join(["?"] * len(columns))
            query = f"INSERT OR {on_conflict} INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"

            with closing(self._get_conn()) as conn:
                conn.execute(query, list(data.values()))
                conn.commit()
                return True

        except sqlite3.Error as e:
            print(f"Insert error: {e}")
            return False

    def bulk_insert(
        self,
        table: str,
        columns: List[str],
        data: List[List[Any]],
        on_conflict: str = "IGNORE"
    ) -> int:
        """
        批量插入資料

        Args:
            table: 資料表名稱
            columns: 欄位名稱列表
            data: 要插入的資料列表 (每筆資料的值列表)
            on_conflict: 衝突處理方式

        Returns:
            成功插入的筆數
        """
        try:
            placeholders = ", ".join(["?"] * len(columns))
            query = f"INSERT OR {on_conflict} INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"

            with closing(self._get_conn()) as conn:
                cursor = conn.cursor()
                cursor.executemany(query, data)
                conn.commit()
                return cursor.rowcount

        except sqlite3.Error as e:
            print(f"Bulk insert error: {e}")
            return 0

    def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: Dict[str, Any]
    ) -> bool:
        """
        執行 UPDATE 操作

        Args:
            table: 資料表名稱
            data: 要更新的資料 {欄位名: 新值}
            where: 條件字典 {欄位名: 值}

        Returns:
            是否成功
        """
        try:
            set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
            where_clause = " AND ".join([f"{k} = ?" for k in where.keys()])
            query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"

            with closing(self._get_conn()) as conn:
                conn.execute(query, list(data.values()) + list(where.values()))
                conn.commit()
                return True

        except sqlite3.Error as e:
            print(f"Update error: {e}")
            return False

    def delete(
        self,
        table: str,
        where: Dict[str, Any]
    ) -> bool:
        """
        執行 DELETE 操作

        Args:
            table: 資料表名稱
            where: 條件字典 {欄位名: 值}

        Returns:
            是否成功
        """
        try:
            where_clause = " AND ".join([f"{k} = ?" for k in where.keys()])
            query = f"DELETE FROM {table} WHERE {where_clause}"

            with closing(self._get_conn()) as conn:
                conn.execute(query, list(where.values()))
                conn.commit()
                return True

        except sqlite3.Error as e:
            print(f"Delete error: {e}")
            return False

    def execute_query(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None
    ) -> List[Dict[str, Any]]:
        """
        執行自定義查詢

        Args:
            query: SQL 查詢語句
            params: 查詢參數

        Returns:
            查詢結果的字典列表
        """
        try:
            with closing(self._get_conn()) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or ())
                return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            print(f"Query error: {e}")
            return []

    def get_company_info(self, stock_id: str) -> Optional[Dict[str, Any]]:
        """獲取單一公司資訊"""
        result = self.select("company_info", where={"stock_id": stock_id}, limit=1)
        return result[0] if result else None

    def insert_daily_data(self, data: Dict[str, Any]) -> bool:
        """插入單日交易數據"""
        required_fields = {"stock_id", "date", "open_price", "high_price",
                          "low_price", "close_price", "volume"}
        if not required_fields.issubset(data.keys()):
            return False
        return self.insert("daily_data", data)

    def bulk_insert_daily_data(self, data: List[Dict[str, Any]]) -> int:
        """批量插入交易數據"""
        if not data:
            return 0

        # 確保所有字典有相同鍵
        columns = list(data[0].keys())
        values = [list(item.values()) for item in data]
        return self.bulk_insert("daily_data", columns, values)