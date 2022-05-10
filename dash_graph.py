import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import time
localtime = time.asctime(time.localtime(time.time()))

dash_app = dash.Dash(__name__)

df = pd.read_csv('samples.csv')
fig = px.line(df, x="Year", y="Value", color='Country Name',
              title=f'Data loaded at {localtime}')
# fig.show()

# this section makes the integration with flask; bootstrap.min.css


def create_dash_application(flask_app):
    dash_app = dash.Dash(server=flask_app, name="graphing",
                         url_base_pathname="/graphing/")

    dash_app.layout = html.Div(
        children=[
            html.A("HOME", href='/'),
            dcc.Graph(
                id='example-graph',
                figure=fig
            )
        ]
    )

    return dash_app


if __name__ == '__main__':

    dash_app.layout = html.Div(
        children=[
            html.H1(children='Dash'),
            html.A("HOME", href='/'),
            html.Div(children='''
                Dash: A web application framework for your data '''),
            dcc.Graph(
                id='example-graph',
                figure=fig
            )
        ])

    dash_app.run_server(debug=True)

    # python __init__.py
