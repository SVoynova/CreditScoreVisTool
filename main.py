# View using: http://127.0.0.1:8050/ in the browser


# Import the necessary components from dash
from dash import Dash, dcc, html

# Create a Dash app instance
app = Dash(__name__)

# Define your app layout using dash components
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'Montreal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

