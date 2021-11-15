from datetime import datetime
from flask import Flask, render_template, url_for, request, redirect
from flask.wrappers import Request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# initialize the database
db = SQLAlchemy(app)

# create table


class TB(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<TB %r>' % self.id


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


@app.route('/about')
def about():
    title = "About"
    return render_template("about.html", title=title)
