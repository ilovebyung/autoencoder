from flask import Flask, render_template, url_for, request, redirect, flash, session
from flask.wrappers import Request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, time
from dateutil import parser
from werkzeug.utils import secure_filename
import os
import concurrent.futures
from dateutil import parser
from utility import allowed_file, InfoForm, UPLOAD_FOLDER, upload_file
from database import app, db, Inspection, Setting
from graphing import create_dash_application


'''
database 
'''
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

'''
dash 
'''
# app = Flask(__name__)
create_dash_application(app)

'''
upload module
'''
app.secret_key = "IOT4M@hle"
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'upload_folder')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    title = "Home Page"
    return render_template("index.html", title=title)


@app.route('/add', methods=["POST", "GET"])
def add():
    if request.method == "POST":
        name = request.form['name']
        value = request.form['value']
        new_row = Inspection(name=name, value=value)
        # commit to database
        try:
            db.session.add(new_row)
            db.session.commit()
            return redirect('/add')
        except:
            return "an error has occured"
    else:
        rows = Inspection.query.order_by(Inspection.date_created)
        return render_template("add.html", rows=rows)


@app.route('/query', methods=["POST", "GET"])
def query():
    title = "query by date"

    rows = Inspection.query.order_by(Inspection.date_created)
    return render_template("query.html", title=title, rows=rows)


@app.route('/update/<int:id>', methods=["POST", "GET"])
def update(id):
    row_to_update = Inspection.query.get_or_404(id)

    if request.method == "POST":
        row_to_update.name = request.form['name']
        try:
            db.session.commit()
            return redirect('/query')
        except:
            return "an error has occured"
    else:
        return render_template("update.html", row_to_update=row_to_update)


@app.route('/delete/<int:id>', methods=["POST", "GET"])
def delete(id):
    row_to_delete = Inspection.query.get_or_404(id)
    try:
        db.session.delete(row_to_delete)
        db.session.commit()
        return redirect('/query')
    except:
        return "an error has occured"


@app.route('/upload', methods=["POST", "GET"])
def upload():
    title = "Copy WAV files from source to target data"
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part', category='error')
            # return redirect(request.url)
            return redirect('/upload')

        files = request.files.getlist('files[]')

        # for file in files:
        #     if file and allowed_file(file.filename):
        #         filename = secure_filename(file.filename)
        #         # multithreding
        #         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(upload_file, files)

        flash('WAV File(s) successfully uploaded')
    return render_template("upload.html", title=title)


@app.route('/date', methods=["POST", "GET"])
def date():
    title = "Select date range"
    # return render_template("date_picker.html", title=title)
    form = InfoForm()
    if form.validate_on_submit():
        session['startdate'] = form.startdate.data
        session['enddate'] = str(datetime.combine(
            form.enddate.data, time.max))  # till the end of day
        return redirect('date_result')
    return render_template('date_picker.html', title=title, form=form)


@app.route('/date_result', methods=["POST", "GET"])
def date_result():
    title = " inspection result"
    startdate = parser.parse(session['startdate'])
    enddate = parser.parse(session['enddate'])
    rows = db.session.query(Inspection).filter(
        Inspection.date_created.between(startdate, enddate))
    return render_template('date_result.html', title=title, rows=rows, startdate=startdate.strftime("%Y-%m-%d %H:%M"), enddate=enddate.strftime("%Y-%m-%d %H:%M"))


@app.route('/setting', methods=["POST", "GET"])
def setting():
    title = "Set Threshold: The most recent value is applied"
    if request.method == "POST":
        row = request.form['value']
        new_row = Setting(value=row)

        if len(row) < 1:
            flash("Type in a valid value", category="error")
            return redirect('/setting')
        # commit to database
        try:
            db.session.add(new_row)
            db.session.commit()
            return redirect('/setting')
        except:
            flash("An unknown error has occurred.", category='error')
            return redirect('/setting')
    else:
        rows = Setting.query.order_by(Setting.date_created)
        return render_template("setting.html", title=title, rows=rows)


@app.route('/delete_setting/<int:id>', methods=["POST", "GET"])
def delete_setting(id):
    row_to_delete = Setting.query.get_or_404(id)
    try:
        db.session.delete(row_to_delete)
        db.session.commit()
        return redirect('/setting')
    except:
        flash("A deletion error has occurred.", category='error')
        return redirect('/setting')


@app.route('/graphing')
def graphing():
    create_dash_application(app)
