import sqlite3
from typing import List, Union, Any, Optional, Tuple


DB_NAME: str = "bookstore.db"  # 可根據需要更換資料庫檔案名稱


class DataBaseTool:
    """
    提供 SQLite 資料庫的基本操作功能（Select、Insert、Update、Delete）
    """

    def __init__(self, db_name: str) -> None:
        """
        初始化資料庫工具

        Args:
            db_name (str): 資料庫檔案名稱
        """
        self.DBNAME: str = db_name

    def DBSelect(
        self,
        table: str,
        column: Union[str, List[str]],
        name: str,
        value: Any
    ) -> List[Tuple[Any, ...]]:
        """
        執行 SELECT 查詢操作

        Args:
            table (str): 資料表名稱
            column (Union[str, List[str]]): 欲查詢的欄位名稱或清單
            name (str): 條件欄位名稱
            value (Any): 條件欄位值

        Returns:
            List[Tuple[Any, ...]]: 查詢結果
        """
        try:
            column_str: str = ", ".join(column) if isinstance(column, list) else column
            query: str = f"SELECT {column_str} FROM {table} WHERE {name} = ?"
            with sqlite3.connect(self.DBNAME) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (value,))
                results: List[Tuple[Any, ...]] = cursor.fetchall()
            return results
        except sqlite3.Error:
            return []

    def DBInsert(
        self,
        table: str,
        columns: List[str],
        values: List[Any]
    ) -> bool:
        """
        執行 INSERT 插入操作

        Args:
            table (str): 資料表名稱
            columns (List[str]): 欲插入的欄位名稱
            values (List[Any]): 欲插入的數值

        Returns:
            bool: 插入成功回傳 True，否則 False
        """
        try:
            column_str: str = ", ".join(columns)
            placeholders: str = ", ".join(["?"] * len(values))
            query: str = f"INSERT INTO {table} ({column_str}) VALUES ({placeholders})"
            with sqlite3.connect(self.DBNAME) as conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()
            return True
        except sqlite3.Error:
            return False

    def DBUpdate(
        self,
        table: str,
        columns: List[str],
        values: List[Any],
        condition_column: str,
        condition_value: Any
    ) -> bool:
        """
        執行 UPDATE 更新操作

        Args:
            table (str): 資料表名稱
            columns (List[str]): 欲更新的欄位名稱
            values (List[Any]): 欲更新的數值
            condition_column (str): 條件欄位名稱
            condition_value (Any): 條件欄位值

        Returns:
            bool: 更新成功回傳 True，否則 False
        """
        try:
            set_clause: str = ", ".join([f"{col} = ?" for col in columns])
            query: str = f"UPDATE {table} SET {set_clause} WHERE {condition_column} = ?"
            with sqlite3.connect(self.DBNAME) as conn:
                cursor = conn.cursor()
                cursor.execute(query, values + [condition_value])
                conn.commit()
            return True
        except sqlite3.Error:
            return False

    def DBDelete(
        self,
        table: str,
        condition_column: str,
        condition_value: Any
    ) -> bool:
        """
        執行 DELETE 刪除操作

        Args:
            table (str): 資料表名稱
            condition_column (str): 條件欄位名稱
            condition_value (Any): 條件欄位值

        Returns:
            bool: 刪除成功回傳 True，否則 False
        """
        try:
            query: str = f"DELETE FROM {table} WHERE {condition_column} = ?"
            with sqlite3.connect(self.DBNAME) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (condition_value,))
                conn.commit()
            return True
        except sqlite3.Error:
            return False
