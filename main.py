import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

# Sample Data
df = px.data.gapminder()
df_2007 = df.query("year==2007")

# Create Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(children=[
    # Left column with logo and credit score info
    html.Div(
        style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top'},
        children=[
            # Logo in the top left corner
            html.Img(
                src="logo.png",  # Replace with the actual filename of your logo
                style={'width': '100px', 'height': '100px'}
            ),
            html.H3('Credit Score Info'),
            html.P("Your credit score is a numerical representation of your creditworthiness. "
                   "It's a measure that helps lenders assess the risk of lending money to you. "
                   "A higher credit score indicates lower credit risk.")
        ]
    ),
    # Right column with plots
    html.Div(
        style={'width': '80%', 'display': 'inline-block'},
        children=[
            # First Plot
            dcc.Graph(
                id='scatter-plot',
                figure=px.scatter(
                    df_2007,
                    x="gdpPercap", y="lifeExp", size="pop", color="continent",
                    log_x=True, size_max=60,
                    title="Gapminder 2007: Scatter Plot"
                )
            ),
            # Second Plot
            dcc.Graph(
                id='surface-plot',
                figure=px.scatter_3d(
                    df_2007,
                    x="gdpPercap", y="lifeExp", z="pop", color="continent",
                    size_max=60,
                    title="Gapminder 2007: 3D Scatter Plot"
                )
            ),
        ]
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8060)

