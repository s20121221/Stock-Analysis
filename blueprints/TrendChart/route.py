from flask import Blueprint, render_template, request, jsonify
from Tool.ohlc_service import query_ohlc
TrendChart_bp = Blueprint(
    'TrendChart', 
    __name__, 
    template_folder='../../templates',
    url_prefix='/TrendChart'
)

@TrendChart_bp.route('/', methods=['GET'])
def index():
    return render_template('TrendChart.html')

@TrendChart_bp.route('/api/ohlc')
def api_ohlc():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': '缺少 symbol 參數'}), 400

    # 這裡用你自己的函式把日 OHLC 撈出來
    df = query_ohlc(symbol)          # <- 請自行實作 / 修改

    if df is None or df.empty:       # 沒資料就回傳空陣列
        return jsonify([])

    data = [
        {
            'date':  row['date'].strftime('%Y-%m-%d'),
            'open':  float(row['open']),
            'high':  float(row['high']),
            'low':   float(row['low']),
            'close': float(row['close'])
        }
        for _, row in df.iterrows()
    ]
    return jsonify(data)