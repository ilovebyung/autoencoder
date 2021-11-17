from flask import Flask, render_template, url_for, request, redirect, flash, session
from flask.wrappers import Request
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms.fields.html5 import DateField
from wtforms.validators import DataRequired
from wtforms import validators, SubmitField
from datetime import datetime
from dateutil import parser
from werkzeug.utils import secure_filename
import os

'''
model module
'''
app = Flask(__name__)
# locate database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# suppress error messages
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# initialize the database
db = SQLAlchemy(app)


class TB(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<TB %r>' % self.id


# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

'''
upload module
'''
app.secret_key = "M@hle123"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'upload_folder')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['wav'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


'''
date_picker module
'''


class InfoForm(FlaskForm):
    startdate = DateField('Start Date', format='%Y-%m-%d',
                          validators=(validators.DataRequired(),))
    enddate = DateField('End Date', format='%Y-%m-%d',
                        validators=(validators.DataRequired(),))
    submit = SubmitField('Submit')


start = "Tue, 16 Nov 2021 00:00:00 GMT"
end = "Tue, 16 Nov 2021 00:00:00 GMT"
dt = parser.parse(start)


@app.route('/')
def index():
    title = "Home Page"
    return render_template("index.html", title=title)


@app.route('/add', methods=["POST", "GET"])
def add():
    if request.method == "POST":
        row = request.form['name']
        new_row = TB(name=row)
        # commit to database
        try:
            db.session.add(new_row)
            db.session.commit()
            return redirect('/add')
        except:
            return "an error has occured"
    else:
        rows = TB.query.order_by(TB.date_created)
        return render_template("add.html", rows=rows)


@app.route('/query', methods=["POST", "GET"])
def query():
    title = "query by date"

    rows = TB.query.order_by(TB.date_created)
    return render_template("query.html", title=title, rows=rows)


@app.route('/update/<int:id>', methods=["POST", "GET"])
def update(id):
    row_to_update = TB.query.get_or_404(id)

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
    row_to_delete = TB.query.get_or_404(id)
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
            flash('No file part')
            # return redirect(request.url)
            return redirect('/upload')

        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        flash('WAV File(s) successfully uploaded')
    return render_template("upload.html", title=title)


@app.route('/date', methods=["POST", "GET"])
def date():
    title = "Select date range"
    # return render_template("date_picker.html", title=title)
    form = InfoForm()
    if form.validate_on_submit():
        session['startdate'] = form.startdate.data
        session['enddate'] = form.enddate.data
        return redirect('date_result')
    return render_template('date_picker.html', title=title, form=form)


@app.route('/date_result', methods=["POST", "GET"])
def date_result():
    title = "date_result"
    startdate = session['startdate']
    enddate = session['enddate']
    return render_template('date_result.html')


@app.route('/about')
def about():
    title = "About"
    return render_template("about.html", title=title)
