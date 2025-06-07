from flask import Blueprint, render_template, request, redirect, url_for, flash, session

Setting_bp = Blueprint(
    'Setting', 
    __name__, 
    template_folder='../../templates',
    url_prefix='/Setting'
)

@Setting_bp.route('/', methods=['GET'])
def index():
    return render_template('Setting.html')