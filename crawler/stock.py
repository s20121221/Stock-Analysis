import os
import sys
import requests
import datetime
from typing import List, Dict, Any
from bs4 import BeautifulSoup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Tool.DataBaseTool import DataBaseTool  # 引入自製資料庫操作工具

db = DataBaseTool("db/stock.db")


def fetch_company_info(stock_id: str) -> Dict[str, Any]:
    """
    爬取個股基本資料（名稱、產業、市值、股本）
    :param stock_id: 股票代號（例如 2330）
    :return: 股票公司資料 dict
    """
    url = f"https://tw.stock.yahoo.com/quote/{stock_id}/profile"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, "html.parser")

        # 取得公司名稱
        name_tag = soup.find("h1", {"class": "Fz(20px)"})
        name = name_tag.text.strip() if name_tag else "N/A"

        # 擷取其他欄位
        table = soup.find("table")
        rows = table.find_all("tr") if table else []

        industry = ""
        capital = 0
        m_value = 0

        for row in rows:
            th = row.find("th")
            td = row.find("td")
            if th and td:
                key = th.text.strip()
                val = td.text.strip().replace(",", "")
                if "產業類別" in key:
                    industry = val
                elif "股本" in key and "億" in val:
                    capital = int(float(val.replace("億", "")) * 10000)
                elif "市值" in key and "億" in val:
                    m_value = int(float(val.replace("億", "")) * 10000)

        return {
            "stock_id": stock_id,
            "name": name,
            "industry": industry,
            "capital": capital,
            "mValue": m_value
        }

    except Exception as e:
        print(f"[Error] 爬取公司資訊失敗：{stock_id} => {e}")
        return {}


def fetch_daily_data(stock_id: str, days: int = 30) -> List[Dict[str, Any]]:
    """
    爬取歷史股價資料（預設近30日）
    :param stock_id: 股票代號
    :param days: 幾天前至今（預設30）
    :return: 每日股價資料列表
    """
    end = int(datetime.datetime.now().timestamp())
    start = int((datetime.datetime.now() - datetime.timedelta(days=days)).timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{stock_id}.TW"
        f"?period1={start}&period2={end}&interval=1d"
    )
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        json_data = res.json()

        result = []

        chart_data = json_data.get("chart", {}).get("result", [])
        if not chart_data:
            raise ValueError("無效資料格式")

        timestamps = chart_data[0]["timestamp"]
        indicators = chart_data[0]["indicators"]["quote"][0]

        for i, ts in enumerate(timestamps):
            date_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            item = {
                "stock_id": stock_id,
                "date": date_str,
                "open_price": indicators["open"][i],
                "high_price": indicators["high"][i],
                "low_price": indicators["low"][i],
                "close_price": indicators["close"][i],
                "volume": indicators["volume"][i]
            }

            # 過濾無效資料
            if all(v is not None for v in item.values()):
                result.append(item)

        return result

    except Exception as e:
        print(f"[Error] 爬取股價資料失敗：{stock_id} => {e}")
        return []


def update_stock_data(stock_id: str, days: int = 30) -> None:
    """
    整合公司資訊與每日股價，並寫入資料庫
    """
    print(f"\n[Info] 開始更新 {stock_id}")

    company = fetch_company_info(stock_id)
    if company:
        db.insert("company_info", company)

    daily = fetch_daily_data(stock_id, days)
    if daily:
        db.bulk_insert_daily_data(daily)

    print(f"[Info]. 完成 {stock_id} 更新\n")

# 呼叫此函示 ---- > app.py
def run_stock_crawler(stock_list, days=365*2):
    for stock_id in stock_list:
        update_stock_data(stock_id, days)
# 測試用
if __name__ == "__main__":
    stock_list = ["2330", "2317", "2603", "2882"]
    run_stock_crawler(stock_list, days=60)
