from flask import Blueprint, render_template, request, redirect, url_for, flash, session

Backtesting_bp = Blueprint(
    'Backtesting', 
    __name__, 
    template_folder='../../templates',
    url_prefix='/Backtesting'
)

@Backtesting_bp.route('/', methods=['GET'])
def index():
    return render_template('Backtesting.html')