# backtesting.py (Flask Blueprint)
import os
import threading
import datetime as dt
from flask import Blueprint, render_template, request, jsonify
from LSTM.Stock_LSTM_Code import StockPredictor
from LSTM.Stock_LSTM_Code import StockPredictor
from flask import flash
from crawler.stock import run_stock_crawler

Backtesting_bp = Blueprint(
    'Backtesting',
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
    url_prefix='/Backtesting'
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', '..', 'db', 'stock.db')
model_dir = os.path.join(BASE_DIR, '..', '..','LSTM','stock_Model')

# 儲存訓練進度
training_progress = {}

# 背景訓練函式
def background_train(task_id, predictor, total_epochs):
    for epoch in range(1, total_epochs + 1):
        predictor.train_one_epoch()
        training_progress[task_id] = int(epoch / total_epochs * 100)
    training_progress[task_id] = 100
    savemodel_dir = os.path.join(BASE_DIR, '..', '..','LSTM','stock_Model',f"{task_id}")
    predictor.save_model(savemodel_dir)
# AJAX 呼叫：開始訓練
@Backtesting_bp.route('/train', methods=['POST'])
def train():
    code = request.json.get('stockCode', '2330')
    run_stock_crawler([code], 365*2)
    # 建立 Predictor
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR,'..',  '..', 'db', 'stock.db')
    EPOCHS = 1000
    pred = StockPredictor(
        db_path=DB_PATH,
        stock_id=code,
        seq_len=30,
        batch_size=32,
        epochs=EPOCHS,
        lr=1e-3,
        model_dir=os.path.join(BASE_DIR, 'stock_Model')
    )
    # 初始化進度
    task_id = code
    training_progress[task_id] = 0
    # 啟動背景執行緒
    thread = threading.Thread(
        target=background_train,
        args=(task_id, pred, EPOCHS),
        daemon=True
    )
    thread.start()
    return jsonify({"task_id": task_id}), 202

# AJAX 輪詢：取得進度
@Backtesting_bp.route('/progress/<task_id>')
def progress(task_id):
    prog = training_progress.get(task_id, 0)
    return jsonify({"progress": prog})

# 主畫面：顯示表單、進度與預測結果
@Backtesting_bp.route('/', methods=['GET', 'POST'])
def index():
    next_date = None
    prediction = None

    if request.method == 'POST':
        action = request.form.get('action')
        code   = request.form.get('stockCode', '2330')
        # 讀取使用者輸入的回測日期（字串）
        period_str = request.form.get('period', '').strip()

        # 回測分析按鈕
        if action == 'backtest':
            predictor = StockPredictor(
                db_path=DB_PATH,
                stock_id=code,
                seq_len=30,
                batch_size=32,
                epochs=0,
                lr=1e-3,
                model_dir=model_dir
            )
            predictor.load_model()

            # 如果有指定 period，就傳入；沒有就用預設（最後一筆日期）
            try:
                if period_str:
                    # predict_next_day 內會把字串轉成 date
                    nd, pred = predictor.predict_next_day(period_str)
                else:
                    nd, pred = predictor.predict_next_day()

                next_date   = nd.strftime('%Y-%m-%d')
                prediction  = f"{pred:.2f}"

            except ValueError as e:
                # 處理傳入不在範圍內或資料不足的情況
                flash(str(e), 'danger')
                next_date = None
                prediction = None

    return render_template(
        'Backtesting.html',
        next_date=next_date,
        prediction=prediction
    )