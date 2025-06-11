import pandas as pd
from typing import Union
import datetime as dt
from Tool.DataBaseTool import DataBaseTool, DB_NAME;

db = DataBaseTool(DB_NAME)   # 先建立一個共用實例

def query_ohlc(symbol: str, start=None, end=None):
    import pandas as pd, datetime as dt, random
    today = dt.date.today()
    rows = []
    price = 100
    for i in range(30):
        d = today - dt.timedelta(days=i)
        open_p = price
        high_p = price + random.uniform(0, 5)
        low_p  = price - random.uniform(0, 5)
        close_p = low_p + random.uniform(0, high_p - low_p)
        rows.append({'date': d, 'open': open_p,
                     'high': high_p, 'low': low_p,
                     'close': close_p})
        price += random.uniform(-2, 2)
    return pd.DataFrame(rows).sort_values('date')

