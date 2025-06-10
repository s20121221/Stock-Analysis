/* static/js/trendchart.js */

async function drawCandlestick(symbol) {
  // 呼叫後端 API 取得 OHLC JSON
  const resp = await fetch(`/TrendChart/api/ohlc?symbol=${encodeURIComponent(symbol)}`);
  if (!resp.ok) {
    alert("API 取資料失敗");         // 基本錯誤提示
    return;
  }
  const data = await resp.json();
  if (!data.length) {
    alert("查無資料");               // 空陣列時提示
    return;
  }

  // 拆欄位
  const dates = data.map(d => d.date);
  const open  = data.map(d => d.open);
  const high  = data.map(d => d.high);
  const low   = data.map(d => d.low);
  const close = data.map(d => d.close);

  // 畫 Plotly Candlestick
  Plotly.newPlot("candlestick-chart", [{
    x: dates, open, high, low, close,
    type: "candlestick",
    increasing: { line: { color: "green" } },
    decreasing: { line: { color: "red" } },
    hoverinfo: "x+open+high+low+close"
  }], {
    title: `${symbol} 日 K 線圖`,
    xaxis: { rangeslider: { visible: false } },
    yaxis: { autorange: true }
  });
}

/* 綁定按鈕事件（在 DOM 全部載好之後） */
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("btnDraw")
    .addEventListener("click", () => {
      const symbol = document.getElementById("symbol").value;
      drawCandlestick(symbol);
    });
});
