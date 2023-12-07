import dash
from dash import dcc, html, Input, Output, ClientsideFunction
import plotly.express as px
import pandas as pd
import pathlib

# Read the CSV file with explicit dtype for the 'Month' column
# Specify the dtype for the 'Month' column and treat other mixed types columns as strings
dtype_mapping = {'Month': 'object', 'Payment_Behaviour': 'str', 'Column2': 'str'}  # Replace with the actual column names

# Read the CSV file with specified dtypes
df = pd.read_csv('all_data.csv', delimiter=';', dtype=dtype_mapping, low_memory=False)

# Map month names to integer values
month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

# Create a new column 'Month_Int' with mapped integer values
df['Month_Int'] = df['Month'].map(month_mapping)

# Print the dataframe
print(df.to_string())

# Initialize the Dash app
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Credit Data Visualization"

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Register all columns for callbacks
all_columns = df.columns.tolist()
scatter_inputs = [Input((i + "_scatter_plot"), "selectedData") for i in all_columns]


def description_card():
    return html.Div(
        id="description-card",
        children=[
            html.H5("Credit Data Analytics"),
            html.H3("Welcome to the Credit Data Visualization Dashboard"),
            html.Div(
                id="intro",
                children="Explore credit data over time. Click on the scatter plot to visualize the relationship between different variables.",
            ),
        ],
    )


def generate_control_card():
    return html.Div(
        id="control-card",
        children=[
            html.P("Select Variable for X-axis"),
            dcc.Dropdown(
                id="x-axis-select",
                options=[{"label": col, "value": col} for col in all_columns],
                value=all_columns[0],
            ),
            html.Br(),
            html.P("Select Variable for Y-axis"),
            dcc.Dropdown(
                id="y-axis-select",
                options=[{"label": col, "value": col} for col in all_columns],
                value=all_columns[1],
            ),
        ],
    )


def generate_scatter_plot(x_axis, y_axis, scatter_click):
    hovertemplate = "<b>Month</b>: %{text} <br><b>%{y}</b>: %{x}"
    scatter_fig = px.scatter(df, x=x_axis, y=y_axis, hover_data=['Month_Int'],
                             title="Credit Score vs Month", labels={'Month_Int': 'Month', 'Credit_Score': 'Credit Score'},
                             hovertemplate=hovertemplate)

    # Highlight clicked point
    if scatter_click is not None:
        selected_index = scatter_click["points"][0]["pointIndex"]
        scatter_fig.data[0].selectedpoints = [selected_index]

    return scatter_fig


app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("plotly_logo.png"))],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
                     + [
                         html.Div(
                             ["initial child"], id="output-clientside", style={"display": "none"}
                         )
                     ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # Scatter Plot
                html.Div(
                    id="scatter_plot_card",
                    children=[
                        html.B("Scatter Plot"),
                        html.Hr(),
                        dcc.Graph(id="scatter_plot"),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("scatter_plot", "figure"),
    [
        Input("x-axis-select", "value"),
        Input("y-axis-select", "value"),
        Input("scatter_plot", "selectedData"),
    ],
)
def update_scatter_plot(x_axis, y_axis, scatter_click):
    return generate_scatter_plot(x_axis, y_axis, scatter_click)


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("scatter_plot", "selectedData")] + scatter_inputs,
)

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
