# crawler/news_crawler.py
import requests
from bs4 import BeautifulSoup
from Tool.DataBaseTool import DataBaseTool
import logging


class News:
    def __init__(self):
        self.db = DataBaseTool()
        # 設定瀏覽器身分
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        }
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    # 抓新聞資料
    def fetch_news(self, stock_id):
        url = f"https://example.com/stock-news/{stock_id}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            news_list = []
            for article in soup.select('.news-item'):
                title = article.select_one('.title').text.strip()
                date = article.select_one('.date').text.strip()
                content = article.select_one('.content').text.strip()

                news_list.append((
                    stock_id,
                    date,
                    title,
                    content,
                    "example.com"  # 來源
                ))

            return news_list
        except Exception as e:
            self.logger.error(f"Failed to fetch news: {e}")
            return []

    # 更新新聞資料庫 ---> news_data
    def update_news(self, stock_id):
        news_data = self.fetch_news(stock_id)
        if not news_data:
            self.logger.warning(f"No news found for {stock_id}")
            return False

        query = """INSERT OR IGNORE INTO news_data
                   (stock_id, date, title, content, source)
                   VALUES (?, ?, ?, ?, ?)"""
        self.db.create_connection()
        self.db.execute_query(query, news_data, many=True)
        self.db.close_connection()
        self.logger.info(f"Updated {len(news_data)} news for {stock_id}")
        return True
