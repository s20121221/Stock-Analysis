from flask import Blueprint, render_template, request, jsonify
from Tool.DataBaseTool import DataBaseTool

TrendChart_bp = Blueprint(
    'TrendChart',
    __name__,
    template_folder='../../templates',
    url_prefix='/TrendChart'
)

db = DataBaseTool("db/stock.db")

@TrendChart_bp.route('/')
def index():
    return render_template("TrendChart.html")


@TrendChart_bp.route('/api/symbols')
def api_symbols():
    results = db.select("company_info", columns="stock_id", order_by="stock_id ASC")
    symbols = [row["stock_id"] for row in results]
    return jsonify(symbols)


@TrendChart_bp.route('/api/ohlc')
def api_ohlc():
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"error": "缺少 symbol 參數"}), 400

    results = db.select(
        table="daily_data",
        columns=["date", "open_price", "high_price", "low_price", "close_price"],
        where={"stock_id": symbol},
        order_by="date ASC"
    )

    if not results:
        return jsonify([])

    data = [
        {
            "date": row["date"],
            "open": row["open_price"],
            "high": row["high_price"],
            "low": row["low_price"],
            "close": row["close_price"]
        }
        for row in results
    ]

    return jsonify(data)