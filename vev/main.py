# RUN THIS
from dash.dependencies import Input, Output, State
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.express as px
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd


selected_features = {}

# Load the dataset
df = pd.read_csv(r'cleaned01.csv', low_memory=False)
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

colorblind_friendly_colors = [
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
]

import colorsys

# Define a base color in RGB
base_color_rgb = [45, 105, 196]  # RGB for #2D69C4


# Function to convert RGB to HSL
def rgb_to_hsl(rgb):
    return colorsys.rgb_to_hls(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)


# Function to convert HSL to RGB
def hsl_to_rgb(hsl):
    return colorsys.hls_to_rgb(hsl[0], hsl[1], hsl[2])


# Base HSL color
base_hsl = rgb_to_hsl(base_color_rgb)

# Generate color shades with different lightness
color_shades = []
for i in range(6):
    # Adjust lightness by 10% increments from 20% to 70%
    new_lightness = 0.2 + (i * 0.1)
    new_color_hsl = (base_hsl[0], new_lightness, base_hsl[2])

    # Convert back to RGB
    new_color_rgb = hsl_to_rgb(new_color_hsl)

    # Format as hex string and append to list
    new_color_hex = '#{:02x}{:02x}{:02x}'.format(
        int(new_color_rgb[0] * 255),
        int(new_color_rgb[1] * 255),
        int(new_color_rgb[2] * 255)
    )
    color_shades.append(new_color_hex)

# Define color mapping for categories
color_mapping = {
    'Low_spent_Small_value_payments': color_shades[0],
    'Low_spent_Medium_value_payments': color_shades[1],
    'Low_spent_Large_value_payments': color_shades[2],
    'High_spent_Medium_value_payments': color_shades[3],
    'High_spent_Large_value_payments': color_shades[4],
    'High_spent_Small_value_payments': color_shades[5]
}

# Set grey for 'No Data'
color_mapping['No Data'] = '#808080'



def find_changed_attributes(df, customer_id):
    changed_attributes = {}

    customer_df = df[df['Customer_ID'] == customer_id].sort_values('Month')

    customer_df['Credit_Score_Changed'] = customer_df['Credit_Score_Num'].diff().ne(0)

    for current_position in range(len(customer_df)):
        current_row = customer_df.iloc[current_position]
        if current_row['Credit_Score_Changed']:
            changes = []
            # Look up to 2 months back, but not before the start of the DataFrame
            start_position = max(0, current_position - 2)
            for prev_position in range(start_position, current_position):
                prev_row = customer_df.iloc[prev_position]
                for col in customer_df.columns:
                    if col not in ['Customer_ID', 'Month', 'Credit_Score', 'Credit_Score_Num', 'Credit_Score_Changed']:
                        if prev_row[col] != current_row[col]:
                            change_desc = f"{col} changed from {prev_row[col]} to {current_row[col]}"
                            changes.append(change_desc)
            # Record the changes
            month_key = current_row['Month'].strftime('%Y-%m') if isinstance(current_row['Month'], pd.Timestamp) else str(current_row['Month'])
            changed_attributes[month_key] = ', '.join(changes) if changes else 'Credit score changed, no other changes detected'
    #print(changed_attributes)
    return changed_attributes

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
# Define colorblind-friendly colors for the pie charts
pie_chart_colors = [
    '#4C78A8',  # blue
    '#F58518',  # orange
    '#54A24B',  # green
    '#E45756',  # red
    '#72B7B2',  # cyan
    '#B279A2',  # purple
    '#FF9DA6',  # pink
    '#9D755D',  # brown
    '#BAB0AC'   # grey
]

base_color = '#2D69C4'  # A shade of blue

# Generate shades by varying the lightness and saturation
color_shades = [
    f"hsl(214, 90%, {percentage}%)"
    for percentage in range(60, 90, 5)  # Start at 60% lightness and increase by 5% each time
]


# Define function to create pie chart with consistent colors
def create_pie_chart(df, title):
    payment_counts = df['Payment_Behaviour'].value_counts()
    payment_counts = payment_counts.sort_index()
    labels = payment_counts.index
    values = payment_counts.values

    fig = px.pie(df, values=values, names=labels, hole=0.3, title=title,
                 color_discrete_sequence=colorblind_friendly_colors)
    if "Poor" in title or "Good" in title or "Standard" in title:
        fig.update_layout(title=title, showlegend=False)
    else:
        fig.update_layout(title=title, showlegend=True)

    # coloring
    fig.update_traces(marker=dict(colors=[color_mapping[label] for label in labels]))
    #fig.update_traces(marker=dict(colors=pie_chart_colors[:len(labels)]))

    return fig


# Create the new plot
def create_new_plot(df):
    # Map numeric credit scores to categories
    df['Credit_Score_Category'] = df['Credit_Score'] #.map({1: 'Poor', 2: 'Standard', 3: 'Good'})

    fig = px.scatter(
        df,
        x="Num_of_Delayed_Payment", y="Delay_from_due_date",
        color="Credit_Score_Category",
        title="Credit Behavior: Payment Delays and Due Date Deviations",
        labels={'Num_of_Delayed_Payment': 'Number of Delayed Payments', 'Delay_from_due_date': 'Delay from Due Date'},
        category_orders={"Credit_Score_Category": ["Poor", "Standard", "Good"]},  # Specify the order of categories
        color_discrete_map={
            'Poor': colorblind_friendly_colors[0],
            'Standard': colorblind_friendly_colors[1],
            'Good': colorblind_friendly_colors[2]
        }
    )
    return fig

# Calculate correlation between Annual Income and Credit Score for each occupation
correlation_results = {}
for occupation in df['Occupation'].unique():
    subset = df[df['Occupation'] == occupation]
    df.loc[df['Occupation'] == occupation, 'Credit_Score_Num'] = df.loc[
        df['Occupation'] == occupation, 'Credit_Score'].map(score_mapping)

# Rename columns for better clarity
result_df = df.groupby(['Occupation', 'Credit_Score']).agg({'Annual_Income': 'median', 'Credit_Score_Num': 'size'}).unstack(fill_value=0)
result_df.columns = [f'{col[0]}_{col[1]}' for col in result_df.columns]

# Reset index to flatten the multi-index
result_df.reset_index(inplace=True)

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Modal definition (Step 1)
instructions_modal = dbc.Modal(
    [
        dbc.ModalBody("Please select all necessary features before clicking 'Save Selection'."),
        dbc.ModalFooter(dbc.Button("Close", id="close-modal", className="ml-auto"))
    ],
    id="instructions-modal",
    size="lg",
    is_open=False,  # Initially not visible
    style={"color": "black"}
)

# Modal for displaying clicked point details
details_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Details"), id="details-modal-header"),
        dbc.ModalBody(id="modal-body-content"),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-details-modal", className="ml-auto")
        ),
    ],
    id="details-modal",
    is_open=False,
)
# Assuming 'occupation_options' is your list of options for the 'occupation-dropdown'
occupation_options = [{'label': occupation if occupation != '______' else 'Unknown', 'value': occupation} for occupation in df['Occupation'].unique()]


# Define the layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                # Left column with logo and credit score info
                dbc.Col(
                    [
                        html.Img(
                            src="/assets/logo.png",
                            style={'width': '100px', 'height': '100px'}
                        ),
                        html.H3(''),
                        html.H2('CreditVis'),
                        html.P(dcc.Markdown('**Your Credit Score Analysis Dashboard**')),
                        html.P(style={'font-size': '12px'}, children="Empower Your Financial Decisions with Data-Driven Insights. "
                               f"CreditVis provides you with an interactive experience to explore and understand the "
                               "factors influencing credit scores. Dive deep into trends, compare scenarios, and uncover "
                               "the nuances of creditworthiness with our comprehensive analysis tool."),
                        html.Hr(),  # Horizontal line for separation
                        html.H5("How to use CreditVis?"),
                        html.P(style={'font-size': '13px'}, children = dcc.Markdown('**1. Select Your Filters:** Begin by choosing your parameters.')),
                        html.P(style={'font-size': '13px'}, children = dcc.Markdown('**2. Save Your Selection:** With a click, save your settings to see the data that matters most to you.')),
                        html.P(style={'font-size': '13px'}, children = dcc.Markdown('**3. Interact with Visuals:** Click on any point in our graphs to get more detailed information. Hover over elements to reveal additional data points.')),
                        html.Hr(),  # Horizontal line for separation
                        # Dropdown menus with labels
                        html.Label('Occupation:'),
                        # Then use 'occupation_options' in your dcc.Dropdown component
                        dcc.Dropdown(
                            id='occupation-dropdown',
                            options=occupation_options,
                            value='All Occupations',  # Set the default value if needed
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
                            value='All incomes',  # Default value
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
                            value='All ages',  # Default value
                            clearable=False  # Prevents users from clearing the selection
                        ),
                        html.Button('Save Selection', id='save-button', n_clicks=0, className='custom-save-button'),
                        # Add this to your app layout
                        html.Div(id='dummy-div', style={'display': 'none'})
                    ],
                    width=3, # Adjust the width as needed
                ),
                # Right column with interactive plot, table, and correlation results
                dbc.Col(
                    [
                        instructions_modal,
                        # Wrap the Graph component with dcc.Loading for lazy loading
                        dcc.Loading(
                            id="loading-trend-graph",
                            children=[dcc.Graph(id='credit-score-trend-graph')],
                            type="default"
                        ),

                        # Add the trend analysis graph
                        #dcc.Graph(id='credit-score-trend-graph'),

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

                        # Row for the credit score specific pie charts
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='good-pie-chart', figure={}), width=4),
                            dbc.Col(dcc.Graph(id='standard-pie-chart', figure={}), width=4),
                            dbc.Col(dcc.Graph(id='poor-pie-chart', figure={}), width=4),
                        ]),
                        details_modal,
                    ],
                    width=9,  # Adjust the width as needed
                ),
            ]
        )
    ],
    fluid=True,
)

# Callback to open modal on button click if clicks are 0
@app.callback(
    Output("instructions-modal", "is_open"),
    [Input("save-button", "n_clicks"), Input("close-modal", "n_clicks")],
    [State("instructions-modal", "is_open")]
)
def toggle_modal(n_clicks_save, n_clicks_close, is_open):
    if n_clicks_save == 0:
        return True
    if n_clicks_close or n_clicks_save:
        return False
    return is_open
'''
@app.callback(
    Output("overlay", "style"),
    [Input("save-button", "n_clicks")],
    [State("overlay", "style")]
)
def toggle_overlay(n_clicks, style):
    if n_clicks and n_clicks > 0:
        style["display"] = "none"  # Hide overlay after button is clicked
    else:
        style["display"] = "block"  # Show overlay initially or when conditions are reset
    return style
'''


@app.callback(
    [Output('details-modal', 'is_open'),
     Output('modal-body-content', 'children')],
    [Input('credit-score-trend-graph', 'clickData'),
     Input('close-details-modal', 'n_clicks')],
    [State('details-modal', 'is_open')]
)
def display_click_data(clickData, btn_n_clicks, is_open):
    if clickData:
        print("click:" , clickData)
        point_info = clickData['points'][0]
        details = point_info.get('customdata', 'No details available')
        print("detals:" , details)
        # Handle case when there are no changes
        if "No changes" in details:
            formatted_details_html = html.P("No changes detected")
        else:
            # Process each detail for formatting
            forms = details.split(", ")
            print(forms)
            formatted_details_html = []
            for form in forms:
                if form[0] in str(formatted_details_html):
                    continue
                # Skip the ID changed lines
                if "ID changed from" in form or "Name changed" in form:
                   continue

                # Replace underscores with spaces
                form = form.replace("_______", "Unknown")
                form = form.replace("_", " ")
                form = form.replace("nan", "Unknown")
                # Replace " _ " with "Unknown"
                form = form.replace(" _ ", " Unknown ")
                # Split each form by space to check for numbers
                parts = form.split()
                new_parts = []
                print(form)
                for part in parts:
                    # If part is a number, format it to two decimal places
                    if part.replace('.', '', 1).isdigit():
                        if "Age" in str(form):
                            part = f"{int(part)}"
                        else:
                            part = f"{float(part):.2f}"
                    new_parts.append(part)
                # Join the parts back and create a paragraph
                new_form = " ".join(new_parts)
                formatted_details_html.append(html.P(new_form))

        return True, formatted_details_html
    elif btn_n_clicks:
        return False, None

    return is_open, None


# Callback to update the trend analysis graph
@app.callback(
    Output('credit-score-trend-graph', 'figure'),
    [Input('save-button', 'n_clicks')],
    [State('occupation-dropdown', 'value'),
     State('income-dropdown', 'value'),
     State('age-dropdown', 'value')]
)
def update_trend_analysis(n_clicks, occupation, income, age):
    # Start with the original DataFrame
    filtered_df = df.copy()

    # Filter based on occupation
    if occupation and occupation != 'All Occupations':
        filtered_df = filtered_df[filtered_df['Occupation'] == occupation]

    # Filter based on income
    if income and income != 'All Incomes':
        # Assuming income dropdown provides a string like '10000-50000'
        if '-' in income:
            income_min, income_max = map(int, income.split('-'))
            filtered_df = filtered_df[
                (filtered_df['Annual_Income'] >= income_min) & (filtered_df['Annual_Income'] <= income_max)]
        else:
            # Handle specific cases like '0' or other non-range values as needed
            pass

    # Filter based on age
    if age and age != 'All Ages':
        # Assuming age dropdown provides a string like '18-25'
        if '-' in age:
            age_min, age_max = map(int, age.split('-'))
            filtered_df = filtered_df[(filtered_df['Age'] >= age_min) & (filtered_df['Age'] <= age_max)]
        else:
            # Handle specific cases like '0' or other non-range values as needed
            pass

    # Proceed with filtered_df for trend analysis...
    trend_data = calculate_trend(filtered_df)

    fig = go.Figure()
    customer_counter = 1  # Initialize a counter for customer numbering

    for customer_id in trend_data['Customer_ID'].unique():
        customer_data = trend_data[trend_data['Customer_ID'] == customer_id]
        changed_attrs = find_changed_attributes(df, customer_id)

        # Create custom data to store the detailed changes
        custom_data = [
            changed_attrs.get(row['Month'].strftime('%Y-%m'), 'No changes')
            for index, row in customer_data.iterrows()
        ]

        # Set the hover text to a simple instruction
        hover_texts = ['Click to view changes' for _ in customer_data.index]

        fig.add_trace(go.Scatter(
            x=customer_data['Month'],
            y=customer_data['Credit_Score_Num'],
            mode='lines+markers',
            name=f'Customer {customer_counter}',
            text=hover_texts,  # This will be displayed on hover
            customdata=custom_data,  # This will store the actual changes but not display them
            hoverinfo='text'  # Only display the text on hover
        ))
        customer_counter += 1  # Increment the counter

    fig.update_layout(title='Credit Score Trend Analysis', xaxis_title='Month', yaxis_title='Credit Score')
    return fig



# Update pie charts
@app.callback(
    [
        Output('good-pie-chart', 'figure'),
        Output('standard-pie-chart', 'figure'),
        Output('poor-pie-chart', 'figure')
    ],
    [Input('save-button', 'n_clicks')],
    [State('occupation-dropdown', 'value'),
     State('income-dropdown', 'value'),
     State('age-dropdown', 'value')]
)
def update_pie_charts(n_clicks, occupation, income, age):
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
        filtered_df = filtered_df[
            (filtered_df['Annual_Income'] >= income_min) & (filtered_df['Annual_Income'] <= income_max)]
    if age != 'All':
        age_min, age_max = parse_age_range(age)
        filtered_df = filtered_df[(filtered_df['Age'] >= age_min) & (filtered_df['Age'] <= age_max)]

    good_fig = create_pie_chart(good_df, 'Good Credit Score Payment Behaviour')
    standard_fig = create_pie_chart(standard_df, 'Standard Credit Score Payment Behaviour')
    poor_fig = create_pie_chart(poor_df, 'Poor Credit Score Payment Behaviour')

    return good_fig, standard_fig, poor_fig

'''
@app.callback(
    Output('overall-pie-chart', 'figure'),
    [Input('save-button', 'n_clicks')],
    # Include other inputs if necessary
)
def update_overall_pie_chart(n_clicks):
    # Filter data and create pie chart
    return create_pie_chart(df, 'Payment_Behaviour, Overall Payment Behaviour')
'''
'''
@app.callback(
    [Output('good-pie-chart', 'figure'),
     Output('standard-pie-chart', 'figure'),
     Output('poor-pie-chart', 'figure')],
    [Input('save-button', 'n_clicks')],
    # Include other inputs if necessary
)
def update_credit_score_pie_charts(n_clicks):
    good_fig = create_pie_chart(good_df, 'Payment_Behaviour, Good Credit Score Payment Behaviour')
    standard_fig = create_pie_chart(standard_df, 'Payment_Behaviour, Standard Credit Score Payment Behaviour')
    poor_fig = create_pie_chart(poor_df, 'Payment_Behaviour, Poor Credit Score Payment Behaviour')

    # Here we update the layout to use a common legend
    for fig in [good_fig, standard_fig, poor_fig]:
        fig.update_layout(showlegend=True)

    # We return all the figures to their respective outputs
    return good_fig, standard_fig, poor_fig
'''

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
        {'label': '300,000 - 400,000', 'value': '300000-400000'},
        {'label': '400,000 - 500,000', 'value': '400000-500000'},
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
        title="The Annual Income and Credit Score Connection",
        labels={'Credit_Score': 'Credit Score', 'Annual_Income': 'Annual Income'},
        category_orders={"Credit_Score": ["Good", "Standard", "Poor"]},
        size_max=30,
        color_discrete_map={
            'Poor': colorblind_friendly_colors[0],
            'Standard': colorblind_friendly_colors[1],
            'Good': colorblind_friendly_colors[2]
        }
    )
    return fig


# Define callback to save selected features
@app.callback(
    dash.dependencies.Output('dummy-div', 'children'),
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
'''

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
'''
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