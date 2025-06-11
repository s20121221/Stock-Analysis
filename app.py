# app.py
from flask import Flask, render_template, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv
load_dotenv()
import os

# ① LINE SDK 初始化（放最上面）
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler      = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

def create_app():
    app = Flask(__name__)

    # ② 保留既有 Blueprint
    from blueprints.Setting.route import Setting_bp
    from blueprints.Backtesting.route import Backtesting_bp
    from blueprints.TrendChart.route import TrendChart_bp

    app.register_blueprint(Setting_bp)
    app.register_blueprint(Backtesting_bp)
    app.register_blueprint(TrendChart_bp)

    # ③ LINE Webhook
    @app.route("/line/webhook", methods=["POST"])
    def line_webhook():
        signature = request.headers.get("X-Line-Signature", "")
        body = request.get_data(as_text=True)
        try:
            handler.handle(body, signature)
        except InvalidSignatureError:
            abort(400)
        return "OK"

    # ④ Echo Handler
    @handler.add(MessageEvent, message=TextMessage)
    def echo(event):
        print("=== USER ID ===", event.source.user_id)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=event.message.text)
        )

    # ⑤ 首頁
    @app.route("/")
    def home():
        return render_template("base.html")

    return app

def push_test():
    user_id = "Ube0a3575661a37ee9fab22890c0361fc"
    line_bot_api.push_message(
        user_id,
        TextSendMessage(text="推播測試成功！")
    )

if __name__ == "__main__":
    if os.getenv("WERKZEUG_RUN_MAIN") == "true":   # 只在子行程執行一次
        push_test()
    create_app().run(port=5000, debug=True)
    
       
   
    
