from flask import Blueprint, render_template, request, redirect, url_for, flash, session

TrendChart_bp = Blueprint(
    'TrendChart', 
    __name__, 
    template_folder='../../templates',
    url_prefix='/TrendChart'
)

@TrendChart_bp.route('/', methods=['GET'])
def index():
    return render_template('TrendChart.html')