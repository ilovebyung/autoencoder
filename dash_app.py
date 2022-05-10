import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd

dash_app = dash.Dash(__name__)

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

# this section makes the integration with flask


def create_dash_app(flask_app):
    dash_app = dash.Dash(server=flask_app, name="Dashboard",
                         url_base_pathname="/dash/")

    dash_app.layout = html.Div(
        children=[
            html.H1(children='Hello Dash'),
            html.Div(children='''
                Dash: A web application framework for your data '''),
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
            html.H1(children='Hello Dash'),
            html.Div(children='''
                Dash: A web application framework for your data '''),
            dcc.Graph(
                id='example-graph',
                figure=fig
            )
        ])

    dash_app.run_server(debug=True)

    # python __init__.py
