from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def Home():
    return render_template("Base.html")

@app.route("/TrendChart")
def TrendChart():
    return render_template("TrendChart.html")

@app.route("/Setting")
def Setting():
    return render_template("Setting.html")

@app.route("/Sigin")
def Sigin():
    return render_template("Sigin.html")

if __name__ == "__main__":
    app.run(debug=True)