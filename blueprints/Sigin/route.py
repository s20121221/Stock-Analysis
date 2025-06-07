from flask import Blueprint, render_template, request, redirect, url_for, flash, session

Sigin_bp = Blueprint(
    'Sigin', 
    __name__, 
    template_folder='../../templates',
    url_prefix='/Sigin'
)

@Sigin_bp.route('/', methods=['GET'])
def index():
    return render_template('Sigin.html')