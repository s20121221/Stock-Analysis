# Tool/DataBaseTool.py
import sqlite3
from pathlib import Path
from typing import List, Union, Any, Tuple

# 預設資料庫名稱（改成 stock.db）
DB_NAME: str = "stock.db"

class DataBaseTool:
    """
    提供 SQLite 資料庫的基本操作功能（Select、Insert、Update、Delete）
    """

    def __init__(self, db_name: str = DB_NAME) -> None:
        """
        初始化資料庫工具
        """
        self.DBNAME: str = db_name

    # ---------- 通用連線 ----------
    def create_connection(self) -> sqlite3.Connection:
        """
        建立並回傳 SQLite 連線；row_factory 設 Row 方便 dict 取值
        """
        conn = sqlite3.connect(self.DBNAME)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------- Select ----------
    def DBSelect(
        self,
        table: str,
        column: Union[str, List[str]],
        name: str,
        value: Any
    ) -> List[Tuple[Any, ...]]:
        try:
            column_str = ", ".join(column) if isinstance(column, list) else column
            query = f"SELECT {column_str} FROM {table} WHERE {name} = ?"
            with self.create_connection() as conn:
                cur = conn.execute(query, (value,))
                return cur.fetchall()
        except sqlite3.Error:
            return []

    # ---------- Insert ----------
    def DBInsert(
        self,
        table: str,
        columns: List[str],
        values: List[Any]
    ) -> bool:
        try:
            col_str = ", ".join(columns)
            placeholders = ", ".join(["?"] * len(values))
            query = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})"
            with self.create_connection() as conn:
                conn.execute(query, values)
                conn.commit()
            return True
        except sqlite3.Error:
            return False

    # ---------- Update ----------
    def DBUpdate(
        self,
        table: str,
        columns: List[str],
        values: List[Any],
        cond_col: str,
        cond_val: Any
    ) -> bool:
        try:
            set_clause = ", ".join([f"{c}=?" for c in columns])
            query = f"UPDATE {table} SET {set_clause} WHERE {cond_col} = ?"
            with self.create_connection() as conn:
                conn.execute(query, values + [cond_val])
                conn.commit()
            return True
        except sqlite3.Error:
            return False

    # ---------- Delete ----------
    def DBDelete(
        self,
        table: str,
        cond_col: str,
        cond_val: Any
    ) -> bool:
        try:
            query = f"DELETE FROM {table} WHERE {cond_col} = ?"
            with self.create_connection() as conn:
                conn.execute(query, (cond_val,))
                conn.commit()
            return True
        except sqlite3.Error:
            return False
        
    # ---------- fetch_data（單筆查詢） ----------
    def fetch_data(
        self,
        table: str,
        column: Union[str, List[str]],
        name: str,
        value: Any
    ):
        """
        回傳第一筆符合條件的紀錄（Row），若無資料回 None。
        """
        column_str = ", ".join(column) if isinstance(column, list) else column
        query = f"SELECT {column_str} FROM {table} WHERE {name} = ? LIMIT 1"
        with self.create_connection() as conn:
            cur = conn.execute(query, (value,))
            return cur.fetchone()

    # ---------- upsert_company_info ----------
    def upsert_company_info(self, info: dict) -> None:
        """
        將 company_info 字典寫入資料庫（若 symbol 已存在則 UPDATE，否則 INSERT）。
        資料庫需有 company_info(symbol TEXT PRIMARY KEY, name TEXT, industry TEXT, ...)
        """
        symbol = info.get("symbol")
        if not symbol:
            return

        cols = ", ".join(info.keys())
        placeholders = ", ".join(["?"] * len(info))
        update_clause = ", ".join([f"{k}=excluded.{k}" for k in info.keys()])

        query = (
            f"INSERT INTO company_info ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(symbol) DO UPDATE SET {update_clause};"
        )

        with self.create_connection() as conn:
            conn.execute(query, tuple(info.values()))
            conn.commit()




# 如果其他模組想快速拿連線，可用這個輔助函式
def get_conn(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or Path(__file__).parent.parent / DB_NAME
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn
