import requests
import pandas as pd
import datetime as dt
from tabulate import tabulate

''' 取得個股日成交資訊 '''

# 輸入股票代號
stock_id = '2330'
# 當日時間
date = dt.date.today().strftime("%Y%m%d")

# 取得證交所網站資料
stock_data = requests.get(
    f'https://www.twse.com.tw/rwd/zh/ \
        afterTrading/STOCK_DAY?date={date}&stockNo={stock_id}',
    verify=False)

json_data = stock_data.json()
df = pd.DataFrame(data=json_data['data'],
                  columns=json_data['fields'])
print(tabulate(df.tail(), headers='keys', tablefmt='pretty', showindex=False))
