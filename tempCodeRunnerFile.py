from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    # 首頁只載入 base.html 以及一個預設的局部區塊
    return render_template("base.html")

if __name__ == "__main__":
    app.run(debug=True)