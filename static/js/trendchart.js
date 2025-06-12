async function drawCandlestick(symbol) {
  const resp = await fetch(`/TrendChart/api/ohlc?symbol=${encodeURIComponent(symbol)}`);
  if (!resp.ok) {
    alert("API 取資料失敗");
    return;
  }
  const data = await resp.json();
  if (!data.length) {
    alert("查無資料");
    return;
  }

  const dates = data.map(d => d.date);
  const open  = data.map(d => d.open);
  const high  = data.map(d => d.high);
  const low   = data.map(d => d.low);
  const close = data.map(d => d.close);

  Plotly.newPlot("candlestick-chart", [{
  x: dates,
  open, high, low, close,
  type: "candlestick",
  increasing: { line: { color: "green" } },
  decreasing: { line: { color: "red" } },
  hovertemplate:
    "<b>%{x}</b><br>" +
    "開盤：%{open}<br>" +
    "最高：%{high}<br>" +
    "最低：%{low}<br>" +
    "收盤：%{close}<br>" +
    "%{customdata}<extra></extra>",
  customdata: close.map((c, i) => {
    if (open[i] < c) return "📈 漲";
    if (open[i] > c) return "📉 跌";
    return "－ 平盤";
  })
}], {
  title: `${symbol} 日 K 線圖`,
  xaxis: { rangeslider: { visible: false } },
  yaxis: { autorange: true },
  margin: { l: 50, r: 50, t: 60, b: 50 },
  autosize: true,
  responsive: true
});
}

document.addEventListener("DOMContentLoaded", async () => {
  const resp = await fetch("/TrendChart/api/symbols");
  const symbols = await resp.json();

  const select = document.getElementById("symbol");
  select.innerHTML = ""; // 清空原有選項

  symbols.forEach(symbol => {
    const opt = document.createElement("option");
    opt.value = symbol;
    opt.textContent = symbol;
    select.appendChild(opt);
  });

  document.getElementById("btnDraw")
    .addEventListener("click", () => {
      const symbol = select.value;
      drawCandlestick(symbol);
    });
});