from flask import Flask, render_template
from blueprints.Setting.route import Setting_bp
from blueprints.Sigin.route import Sigin_bp
from blueprints.TrendChart.route import TrendChart_bp

def create_app():
    app = Flask(__name__)
    # 註冊 Blueprint
    app.register_blueprint(Setting_bp)
    app.register_blueprint(Sigin_bp)
    app.register_blueprint(TrendChart_bp)
    
    @app.route('/')
    def home():
        return render_template('base.html')
    return app

if __name__ == "__main__":
    create_app().run(debug=True)