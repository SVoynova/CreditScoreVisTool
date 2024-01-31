# RUN THIS
from dash.dependencies import Input, Output, State
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
import pandas as pd
import dash_table
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objs as go

selected_features = {}

# Load the dataset
df = pd.read_csv(r'C:\Users\dgons\Desktop\CreditScoreVisTool-main\vev\cleaned01.csv', low_memory=False)
print(df.columns)

# Replace invalid data with a default value
df['Payment_Behaviour'].replace('!@9#%8', 'No Data', inplace=True)

# Define score mapping
score_mapping = {'Poor': 1, 'Standard': 2, 'Good': 3}

# Map categorical values to numerical values
df['Credit_Score_Num'] = df['Credit_Score'].map(score_mapping)

# Filter data based on credit score
good_df = df[df['Credit_Score'] == 'Good']
standard_df = df[df['Credit_Score'] == 'Standard']
poor_df = df[df['Credit_Score'] == 'Poor']

# Convert 'Month' column to datetime format
df['Month'] = pd.to_datetime(df['Month'], format='%m')

# Function to calculate trend for each individual over time
def calculate_trend(df):
    # Assuming 'Month' is in datetime format
    trend_data = df.groupby(['Month', 'Customer_ID'])['Credit_Score_Num'].mean().reset_index()
    return trend_data

def exclude_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Exclude outliers based on IQR
    df_filtered = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
    return df_filtered

# Define consistent colors for the pie charts
pie_chart_colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600', '#808080']

# Define function to create pie chart with consistent colors
def create_pie_chart(df, title):
    payment_counts = df['Payment_Behaviour'].value_counts()
    payment_counts = payment_counts.sort_index()
    labels = payment_counts.index
    values = payment_counts.values

    fig = px.pie(df, values=values, names=labels, hole=0.3)
    fig.update_layout(title=title, showlegend=True)
    fig.update_traces(marker=dict(colors=pie_chart_colors[:len(labels)]))

    return fig

# Create the new plot
def create_new_plot(df):
    # Map numeric credit scores to categories
    df['Credit_Score_Category'] = df['Credit_Score'] #.map({1: 'Poor', 2: 'Standard', 3: 'Good'})

    fig = px.scatter(
        df,
        x="Num_of_Delayed_Payment", y="Delay_from_due_date",
        color="Credit_Score_Category",
        title="Num_of_Delayed_Payment vs. Delay_from_due_date by Credit Score",
        labels={'Num_of_Delayed_Payment': 'Number of Delayed Payments', 'Delay_from_due_date': 'Delay from Due Date'},
        category_orders={"Credit_Score_Category": ["Poor", "Standard", "Good"]},  # Specify the order of categories
        color_discrete_map = {'Poor': 'red', 'Standard': 'blue', 'Good': 'green'}
    )
    return fig

# Calculate correlation between Annual Income and Credit Score for each occupation
correlation_results = {}
for occupation in df['Occupation'].unique():
    subset = df[df['Occupation'] == occupation]
    df.loc[df['Occupation'] == occupation, 'Credit_Score_Num'] = df.loc[
        df['Occupation'] == occupation, 'Credit_Score'].map(score_mapping)
    correlation = subset[['Annual_Income', 'Credit_Score_Num']].corr().iloc[0, 1]
    correlation_results[occupation] = correlation

# Rename columns for better clarity
result_df = df.groupby(['Occupation', 'Credit_Score']).agg({'Annual_Income': 'median', 'Credit_Score_Num': 'size'}).unstack(fill_value=0)
result_df.columns = [f'{col[0]}_{col[1]}' for col in result_df.columns]

# Reset index to flatten the multi-index
result_df.reset_index(inplace=True)

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
# Define the layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                # Left column with logo and credit score info
                dbc.Col(
                    [
                        html.Img(
                            src="your_logo.png",  # Replace with the actual filename of your logo
                            style={'width': '100px', 'height': '100px'}
                        ),
                        html.H3('Credit Score Info'),
                        html.P("Your credit score is a numerical representation of your creditworthiness. "
                               "It's a measure that helps lenders assess the risk of lending money to you. "
                               "A higher credit score indicates lower credit risk."),
                        html.Hr(),  # Horizontal line for separation
                        # Dropdown menus with labels
                        html.Label('Occupation:'),
                        dcc.Dropdown(
                            id='occupation-dropdown',
                            options=[{'label': occupation, 'value': occupation} for occupation in df['Occupation'].unique()],
                            value='Occupation',  # Default value
                            clearable=False  # Prevents users from clearing the selection
                        ),
                        html.Label('Annual Income:'),
                        dcc.Dropdown(
                            id='income-dropdown',
                            options=[
                                {'label': '0', 'value': '0'},
                                {'label': '0 - 10,000', 'value': '0-10000'},
                                {'label': '10,000 - 50,000', 'value': '10000-50000'},
                                {'label': '50,000 - 100,000', 'value': '50000-100000'},
                                {'label': '100,000 - 150,000', 'value': '100000-150000'},
                                {'label': '150,000 - 200,000', 'value': '150000-200000'},
                                {'label': '200,000 - 250,000', 'value': '200000-250000'},
                                {'label': '250,000 - 300,000', 'value': '250000-300000'},
                                {'label': '300,000 - 350,000', 'value': '300000-350000'},
                                # Add more categories as needed
                                # Example: {'label': '350,000 - 400,000', 'value': '350000-400000'},
                            ],
                            value='0',  # Default value
                            clearable=False  # Prevents users from clearing the selection
                        ),
                        html.Label('Age:'),
                        dcc.Dropdown(
                            id='age-dropdown',
                            options=[
                                {'label': '18-25', 'value': '18-25'},
                                {'label': '26-30', 'value': '26-30'},
                                {'label': '31-40', 'value': '31-40'},
                                {'label': '41-50', 'value': '41-50'},
                                {'label': '51-60', 'value': '51-60'},
                                {'label': '61-70', 'value': '61-70'},
                                {'label': '71-80', 'value': '71-80'},
                                {'label': '81-90', 'value': '81-90'},
                                {'label': '91-100', 'value': '91-100'},
                                {'label': '101-110', 'value': '101-110'}
                            ],
                            value='18-25',  # Default value
                            clearable=False  # Prevents users from clearing the selection
                        ),
                        html.Button('Save Selection', id='save-button', n_clicks=0),
                    ],
                    width=3,  # Adjust the width as needed
                ),
                # Right column with interactive plot, table, and correlation results
                dbc.Col(
                    [
                        # Add the trend analysis graph
                        dcc.Graph(id='credit-score-trend-graph'),

                        # New Graph
                        dcc.Graph(
                            id='new-plot',
                            config={'scrollZoom': False},
                        ),
                        # Interactive Scatter Plot (2nd plot)
                        dcc.Graph(
                            id='interactive-plot2',
                            config={'scrollZoom': False},
                        ),
                        # Pie chart for payment behavior
                        dcc.Graph(
                            id='pie-chart',
                            config={'scrollZoom': False},
                        ),
                        # Additional pie charts for different credit score categories
                        dcc.Graph(
                            id='good-pie-chart',
                            config={'scrollZoom': False},
                        ),
                        dcc.Graph(
                            id='standard-pie-chart',
                            config={'scrollZoom': False},
                        ),
                        dcc.Graph(
                            id='poor-pie-chart',
                            config={'scrollZoom': False},
                        ),
                        # Table
                        dash_table.DataTable(
                            id='table',
                            columns=[
                                {"name": col, "id": col, "presentation": "dropdown"}
                                for col in result_df.columns
                            ],
                            data=result_df.to_dict('records'),
                            style_table={'height': '300px', 'overflowY': 'auto'}
                        ),
                        # Correlation Results
                        html.Div(
                            id='correlation-results',
                            children=[
                                html.H3("Correlation Results"),
                                html.Pre("\n".join([f"Correlation for {occupation}: {correlation}" for occupation, correlation in correlation_results.items()]))
                            ]
                        )
                    ],
                    width=9,  # Adjust the width as needed
                ),
            ]
        )
    ],
    fluid=True,
)

# Callback to update the trend analysis graph
@app.callback(
    Output('credit-score-trend-graph', 'figure'),
    [Input('save-button', 'n_clicks')],
    [State('occupation-dropdown', 'value'),
     State('income-dropdown', 'value'),
     State('age-dropdown', 'value')]
)

def update_trend_analysis(n_clicks, occupation, income, age):
    # Filter the data based on the selected features
    filtered_df = df.copy()
    if occupation != 'All':
        filtered_df = filtered_df[filtered_df['Occupation'] == occupation]
    if income != 'All':
        income_min, income_max = parse_income_range(income)
        filtered_df = filtered_df[
            (filtered_df['Annual_Income'] >= income_min) & (filtered_df['Annual_Income'] <= income_max)]
    if age != 'All':
        age_min, age_max = parse_age_range(age)
        filtered_df = filtered_df[(filtered_df['Age'] >= age_min) & (filtered_df['Age'] <= age_max)]
    
    # Calculate trend analysis
    trend_data = calculate_trend(filtered_df)

    trend_data = trend_data.groupby('Customer_ID').filter(lambda x: x['Credit_Score_Num'].nunique() > 1)

    # Plot trend analysis
    fig = go.Figure()
    for customer_id in trend_data['Customer_ID'].unique():
        customer_trend = trend_data[trend_data['Customer_ID'] == customer_id]
        hover_text = [
            (f"Customer ID: {row['Customer_ID']}<br>"
             f"Month: {row['Month']}<br>"
             f"Annual Income: {df[df['Customer_ID'] == customer_id]['Annual_Income'].iloc[0]}<br>"
             f"Age: {df[df['Customer_ID'] == customer_id]['Age'].iloc[0]}<br>"
             f"Num Bank Accounts: {df[df['Customer_ID'] == customer_id]['Num_Bank_Accounts'].iloc[0]}<br>"
             f"Num of Delayed Payments: {df[df['Customer_ID'] == customer_id]['Num_of_Delayed_Payment'].iloc[0]}<br>"
             f"Credit History Age: {df[df['Customer_ID'] == customer_id]['Credit_History_Age'].iloc[0]}")
            for index, row in customer_trend.iterrows()]
        fig.add_trace(go.Scatter(x=customer_trend['Month'], y=customer_trend['Credit_Score_Num'], mode='lines',
                                 name=f'Customer {customer_id}', hovertext=hover_text))

    # Update layout
    fig.update_layout(
        title='Credit Score Trend Analysis',
        xaxis_title='Month',
        yaxis_title='Credit Score',
        hovermode='closest'  # Show hover labels for closest point
    )
    fig.update_yaxes(tickvals=[1, 2, 3], ticktext=['Poor', 'Standard', 'Good'])

    return fig


# Define callback to update dropdown options dynamically
@app.callback(
    Output('occupation-dropdown', 'options'),
    [Input('occupation-dropdown', 'value')]
)
def update_occupation_options(selected_value):
    options = [{'label': 'All Occupations', 'value': 'All'}]  # Add an option for all occupations
    options += [{'label': occupation, 'value': occupation} for occupation in df['Occupation'].unique()]
    return options

@app.callback(
    Output('income-dropdown', 'options'),
    [Input('income-dropdown', 'value')]
)
def update_income_options(selected_value):
    options = [{'label': 'All Incomes', 'value': 'All'}]  # Add an option for all incomes
    options += [
        {'label': '0', 'value': '0'},
        {'label': '0 - 10,000', 'value': '0-10000'},
        {'label': '10,000 - 50,000', 'value': '10000-50000'},
        {'label': '50,000 - 100,000', 'value': '50000-100000'},
        {'label': '100,000 - 150,000', 'value': '100000-150000'},
        {'label': '150,000 - 200,000', 'value': '150000-200000'},
        {'label': '200,000 - 250,000', 'value': '200000-250000'},
        {'label': '250,000 - 300,000', 'value': '250000-300000'},
        {'label': '300,000 - 350,000', 'value': '300000-350000'},
        # Add more categories as needed
        # Example: {'label': '350,000 - 400,000', 'value': '350000-400000'},
    ]
    return options

# Update the options for the age dropdown menu
@app.callback(
    Output('age-dropdown', 'options'),
    [Input('age-dropdown', 'value')]
)
def update_age_options(selected_value):
    options = [{'label': 'All Ages', 'value': 'All'}]  # Add an option for all ages
    options += [
        {'label': '18-25', 'value': '18-25'},
        {'label': '26-30', 'value': '26-30'},
        {'label': '31-40', 'value': '31-40'},
        {'label': '41-50', 'value': '41-50'},
        {'label': '51-60', 'value': '51-60'},
        {'label': '61-70', 'value': '61-70'},
        {'label': '71-80', 'value': '71-80'},
        {'label': '81-90', 'value': '81-90'},
        {'label': '91-100', 'value': '91-100'},
        {'label': '101-110', 'value': '101-110'}
    ]
    return options


# Define callback to update the 2nd plot based on user input and last saved selection
@app.callback(
    Output('interactive-plot2', 'figure'),
    [Input('save-button', 'n_clicks')],
    [State('occupation-dropdown', 'value'),
     State('income-dropdown', 'value'),
     State('age-dropdown', 'value')]
)
def update_plot2(n_clicks, occupation, income, age):
    global selected_features

    if selected_features:
        occupation = selected_features.get('Occupation')
        income = selected_features.get('Annual Income')
        age = selected_features.get('Age')

    # Filter the data based on the last saved selection
    filtered_df = df.copy()
    if occupation != 'All':
        filtered_df = filtered_df[filtered_df['Occupation'] == occupation]
    if income != 'All':
        income_min, income_max = parse_income_range(income)
        filtered_df = filtered_df[(filtered_df['Annual_Income'] >= income_min) & (filtered_df['Annual_Income'] <= income_max)]
    if age != 'All':
        age_min, age_max = parse_age_range(age)
        filtered_df = filtered_df[(filtered_df['Age'] >= age_min) & (filtered_df['Age'] <= age_max)]

    # Exclude outliers using IQR
    filtered_df['Annual_Income'] = pd.to_numeric(filtered_df['Annual_Income'], errors='coerce')
    filtered_df = exclude_outliers(filtered_df, 'Annual_Income')
    filtered_df = exclude_outliers(filtered_df, 'Num_of_Delayed_Payment')

    # Create the plot
    fig = px.scatter(
        filtered_df,
        x="Annual_Income", y="Credit_Score",
        color="Occupation",
        title="Interactive Credit Score vs. Annual Income",
        labels={'Credit_Score': 'Credit Score', 'Annual_Income': 'Annual Income'},
        category_orders={"Credit_Score": ["Good", "Standard", "Poor"]},
        size_max=30,
        color_discrete_map={'Poor': 'red', 'Standard': 'purple', 'Good': 'green'},  # Change colors here
    )
    return fig


# Define callback to save selected features
@app.callback(
    Output('correlation-results', 'children'),
    [Input('save-button', 'n_clicks')],
    [State('occupation-dropdown', 'value'),
     State('income-dropdown', 'value'),
     State('age-dropdown', 'value')]
)
def save_selected_features(n_clicks, occupation, income, age):
    global selected_features
    if n_clicks > 0:
        selected_features = {'Occupation': occupation, 'Annual Income': income, 'Age': age}
        return f"Selected features saved: {selected_features}"
    else:
        return ""


# Define callback to update the new plot based on user input and selected features
@app.callback(
    Output('new-plot', 'figure'),
    [Input('save-button', 'n_clicks')],
    [State('occupation-dropdown', 'value'),
     State('income-dropdown', 'value'),
     State('age-dropdown', 'value')]
)
def update_new_plot(n_clicks, occupation, income, age):
    global selected_features

    # Use the selected features if available, otherwise use defaults
    if selected_features:
        occupation = selected_features.get('Occupation')
        income = selected_features.get('Annual Income')
        age = selected_features.get('Age')

    # Filter the data based on the selected features
    filtered_df = df.copy()
    if occupation != 'All':
        filtered_df = filtered_df[filtered_df['Occupation'] == occupation]
    if income != 'All':
        income_min, income_max = parse_income_range(income)
        filtered_df = filtered_df[(filtered_df['Annual_Income'] >= income_min) & (filtered_df['Annual_Income'] <= income_max)]
    if age != 'All':
        age_min, age_max = parse_age_range(age)
        filtered_df = filtered_df[(filtered_df['Age'] >= age_min) & (filtered_df['Age'] <= age_max)]

    # Exclude outliers using IQR
    filtered_df['Annual_Income'] = pd.to_numeric(filtered_df['Annual_Income'], errors='coerce')
    filtered_df = exclude_outliers(filtered_df, 'Annual_Income')
    filtered_df = exclude_outliers(filtered_df, 'Num_of_Delayed_Payment')

    # Create the new plot
    fig = create_new_plot(filtered_df)
    return fig


# Define callback to update the pie chart based on user input and selected features
@app.callback(
    Output('pie-chart', 'figure'),
    [Input('save-button', 'n_clicks')],
    [State('occupation-dropdown', 'value'),
     State('income-dropdown', 'value'),
     State('age-dropdown', 'value')]
)
def update_pie_chart(n_clicks, occupation, income, age):
    # Use the selected features if available, otherwise use defaults
    if selected_features:
        occupation = selected_features.get('Occupation')
        income = selected_features.get('Annual Income')
        age = selected_features.get('Age')

    # Filter the data based on the selected features
    filtered_df = df.copy()
    if occupation != 'All':
        filtered_df = filtered_df[filtered_df['Occupation'] == occupation]
    if income != 'All':
        income_min, income_max = parse_income_range(income)
        filtered_df = filtered_df[(filtered_df['Annual_Income'] >= income_min) & (filtered_df['Annual_Income'] <= income_max)]
    if age != 'All':
        age_min, age_max = parse_age_range(age)
        filtered_df = filtered_df[(filtered_df['Age'] >= age_min) & (filtered_df['Age'] <= age_max)]

    # Create the pie chart
    fig = create_pie_chart(filtered_df, 'Payment Behaviour')
    return fig


# Define callback to update the good credit score pie chart based on user input and selected features
@app.callback(
    Output('good-pie-chart', 'figure'),
    [Input('save-button', 'n_clicks')],
    [State('occupation-dropdown', 'value'),
     State('income-dropdown', 'value'),
     State('age-dropdown', 'value')]
)
def update_good_pie_chart(n_clicks, occupation, income, age):
    # Use the selected features if available, otherwise use defaults
    if selected_features:
        occupation = selected_features.get('Occupation')
        income = selected_features.get('Annual Income')
        age = selected_features.get('Age')

    # Filter the data based on the selected features
    filtered_df = good_df.copy()
    if occupation != 'All':
        filtered_df = filtered_df[filtered_df['Occupation'] == occupation]
    if income != 'All':
        income_min, income_max = parse_income_range(income)
        filtered_df = filtered_df[(filtered_df['Annual_Income'] >= income_min) & (filtered_df['Annual_Income'] <= income_max)]
    if age != 'All':
        age_min, age_max = parse_age_range(age)
        filtered_df = filtered_df[(filtered_df['Age'] >= age_min) & (filtered_df['Age'] <= age_max)]

    # Create the pie chart for good credit score
    fig = create_pie_chart(filtered_df, 'Payment Behaviour for Good Credit Score')
    return fig

# Define callback to update the standard credit score pie chart based on user input and selected features
@app.callback(
    Output('standard-pie-chart', 'figure'),
    [Input('save-button', 'n_clicks')],
    [State('occupation-dropdown', 'value'),
     State('income-dropdown', 'value'),
     State('age-dropdown', 'value')]
)
def update_standard_pie_chart(n_clicks, occupation, income, age):
    # Use the selected features if available, otherwise use defaults
    if selected_features:
        occupation = selected_features.get('Occupation')
        income = selected_features.get('Annual Income')
        age = selected_features.get('Age')

    # Filter the data based on the selected features
    filtered_df = standard_df.copy()
    if occupation != 'All':
        filtered_df = filtered_df[filtered_df['Occupation'] == occupation]
    if income != 'All':
        income_min, income_max = parse_income_range(income)
        filtered_df = filtered_df[(filtered_df['Annual_Income'] >= income_min) & (filtered_df['Annual_Income'] <= income_max)]
    if age != 'All':
        age_min, age_max = parse_age_range(age)
        filtered_df = filtered_df[(filtered_df['Age'] >= age_min) & (filtered_df['Age'] <= age_max)]

    # Create the pie chart for standard credit score
    fig = create_pie_chart(filtered_df, 'Payment Behaviour for Standard Credit Score')
    return fig

# Define callback to update the poor credit score pie chart based on user input and selected features
@app.callback(
    Output('poor-pie-chart', 'figure'),
    [Input('save-button', 'n_clicks')],
    [State('occupation-dropdown', 'value'),
     State('income-dropdown', 'value'),
     State('age-dropdown', 'value')]
)
def update_poor_pie_chart(n_clicks, occupation, income, age):
    # Use the selected features if available, otherwise use defaults
    if selected_features:
        occupation = selected_features.get('Occupation')
        income = selected_features.get('Annual Income')
        age = selected_features.get('Age')

    # Filter the data based on the selected features
    filtered_df = poor_df.copy()
    if occupation != 'All':
        filtered_df = filtered_df[filtered_df['Occupation'] == occupation]
    if income != 'All':
        income_min, income_max = parse_income_range(income)
        filtered_df = filtered_df[(filtered_df['Annual_Income'] >= income_min) & (filtered_df['Annual_Income'] <= income_max)]
    if age != 'All':
        age_min, age_max = parse_age_range(age)
        filtered_df = filtered_df[(filtered_df['Age'] >= age_min) & (filtered_df['Age'] <= age_max)]

    # Create the pie chart for poor credit score
    fig = create_pie_chart(filtered_df, 'Payment Behaviour for Poor Credit Score')
    return fig

def parse_income_range(income_range):
    if income_range == '0':
        return 0, 0
    elif income_range == '0-10000':
        return 0, 10000
    elif income_range == '10000-50000':
        return 10000, 50000
    elif income_range == '50000-100000':
        return 50000, 100000
    elif income_range == '100000-150000':
        return 100000, 150000
    elif income_range == '150000-200000':
        return 150000, 200000
    elif income_range == '250000-100000000':
        return 250000, 100000000  # For incomes over $250,000
    else:
        return 0, 0  # Default values


def parse_age_range(age_range):
    # Implement parsing logic for age ranges
    # Example implementation:
    age_ranges = {
        '18-25': (18, 25),
        '26-30': (26, 30),
        '31-39': (31, 39),
        '41-49': (41, 49),
        '51-59': (51, 59),
        '61-69': (61, 69),
        '71-79': (71, 79),
        # Add more age range parsing logic as needed
    }
    return age_ranges.get(age_range, (0, 0))  # Default values for invalid range


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8060)