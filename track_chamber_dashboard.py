import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Track Chamber Analysis Dashboard"

# Load and prepare data at startup
def load_initial_data():
    """Load and prepare the processed data"""
    print("Loading data for dashboard...")
    
    # Read the processed CSV file
    df = pd.read_csv('Book1_processed.csv')
    
    # Convert time columns to datetime
    time_cols = ['INTRO_DATE', 'START_DATE', 'END_DATE']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate processing time in minutes
    df['PROCESSING_TIME_MINUTES'] = (df['END_DATE'] - df['START_DATE']).dt.total_seconds() / 60
    
    # Calculate total time (processing + wait) if CHAMBER_WAIT_DURATION exists
    if 'CHAMBER_WAIT_DURATION' in df.columns:
        df['TOTAL_TIME_MINUTES'] = df['PROCESSING_TIME_MINUTES'] + df['CHAMBER_WAIT_DURATION']
        df['HAS_WAIT_DATA'] = True
    else:
        df['TOTAL_TIME_MINUTES'] = df['PROCESSING_TIME_MINUTES']
        df['HAS_WAIT_DATA'] = False
    
    # Create chamber type from first 3 characters of CHAMBER
    df['CHAMBER_TYPE'] = df['CHAMBER'].str[:3]
    
    # Remove rows with invalid CHAMBER_TYPE (NaN values)
    df = df.dropna(subset=['CHAMBER_TYPE'])
    
    # Remove any invalid processing times
    df = df[df['PROCESSING_TIME_MINUTES'] > 0]
    
    return df

# Load data
initial_df = load_initial_data()

def filter_data(df, selected_recipe, selected_chambers):
    """Filter data based on selections"""
    
    # Filter by recipe
    if selected_recipe:
        df = df[df['TRACK_RCP'] == selected_recipe]
    
    # Filter by chamber types
    if selected_chambers:
        df = df[df['CHAMBER_TYPE'].isin(selected_chambers)]
    
    return df

def calculate_chamber_stats(df):
    """Calculate chamber type statistics including wait times if available"""
    if len(df) == 0:
        return pd.DataFrame()
    
    # Check if this filtered data has wait time information
    has_wait_data = 'CHAMBER_WAIT_DURATION' in df.columns and df['HAS_WAIT_DATA'].iloc[0]
    
    # Basic processing time stats
    stats = df.groupby('CHAMBER_TYPE').agg({
        'PROCESSING_TIME_MINUTES': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'CHAMBER': 'nunique'
    }).round(2)
    
    stats.columns = ['Count', 'Mean_Minutes', 'Median_Minutes', 'Std_Minutes', 
                    'Min_Minutes', 'Max_Minutes', 'Unique_Chambers']
    
    # Add wait time and total time stats if available
    if has_wait_data:
        wait_stats = df.groupby('CHAMBER_TYPE').agg({
            'CHAMBER_WAIT_DURATION': ['mean', 'median', 'std'],
            'TOTAL_TIME_MINUTES': ['mean', 'median', 'std']
        }).round(2)
        
        wait_stats.columns = ['Wait_Mean', 'Wait_Median', 'Wait_Std',
                             'Total_Mean', 'Total_Median', 'Total_Std']
        
        # Merge with main stats
        stats = stats.join(wait_stats)
        
        # Calculate efficiency (processing time / total time)
        stats['Efficiency_Percent'] = (stats['Mean_Minutes'] / stats['Total_Mean'] * 100).round(1)
    
    stats = stats.sort_values('Mean_Minutes', ascending=False)
    
    return stats.reset_index()

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Track Chamber Analysis Dashboard"

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏭 Track Chamber Analysis Dashboard", 
                style={'text-align': 'center', 'color': '#2c3e50', 'margin-bottom': '30px'}),
        html.Hr(style={'border': '2px solid #3498db'})
    ]),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label("📋 Select TRACK_RCP Recipe:", 
                      style={'font-weight': 'bold', 'margin-bottom': '10px'}),
            dcc.Dropdown(
                id='recipe-dropdown',
                options=[{'label': recipe, 'value': recipe} for recipe in sorted(initial_df['TRACK_RCP'].unique())],
                value=sorted(initial_df['TRACK_RCP'].unique())[0],
                placeholder="Select a TRACK_RCP recipe...",
                style={'margin-bottom': '20px'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.Label("🔧 Select Chamber Types:", 
                      style={'font-weight': 'bold', 'margin-bottom': '10px'}),
            dcc.Dropdown(
                id='chamber-dropdown',
                options=[{'label': chamber, 'value': chamber} for chamber in sorted(initial_df['CHAMBER_TYPE'].unique())],
                value=sorted(initial_df['CHAMBER_TYPE'].unique()),
                placeholder="Select chamber types to include...",
                multi=True,
                style={'margin-bottom': '20px'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '4%'})
    ], style={'margin-bottom': '30px', 'padding': '20px', 'background-color': '#f8f9fa', 'border-radius': '10px'}),
    
    # Summary Statistics
    html.Div(id='summary-stats', style={'margin-bottom': '30px'}),
    
    # Charts Layout
    html.Div([
        # Row 1: Performance and Variability Charts
        html.Div([
            html.Div([
                dcc.Graph(id='performance-chart')
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='variability-chart')
            ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%'})
        ]),
        
        # Row 2: Utilization and Distribution Charts  
        html.Div([
            html.Div([
                dcc.Graph(id='utilization-chart')
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='distribution-chart')
            ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%'})
        ], style={'margin-top': '20px'}),
        
        # Row 3: Individual Chamber Analysis
        html.Div([
            dcc.Graph(id='individual-chamber-chart')
        ], style={'margin-top': '20px'})
    ])
], style={'margin': '20px', 'font-family': 'Arial, sans-serif'})

# Callback to populate dropdown options
@app.callback(
    [Output('summary-stats', 'children'),
     Output('performance-chart', 'figure'),
     Output('variability-chart', 'figure'),
     Output('utilization-chart', 'figure'),
     Output('distribution-chart', 'figure'),
     Output('individual-chamber-chart', 'figure')],
    [Input('recipe-dropdown', 'value'),
     Input('chamber-dropdown', 'value')]
)
def update_charts(selected_recipe, selected_chambers):
    if not selected_recipe or not selected_chambers:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Please select a recipe and chamber types", 
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return html.Div(), empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
    
    # Filter data
    df = filter_data(initial_df.copy(), selected_recipe, selected_chambers)
    
    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available for current selection", 
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return html.Div(), empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
    
    # Calculate statistics
    chamber_stats = calculate_chamber_stats(df)
    
    # Create summary statistics
    total_wafers = len(df)
    chamber_types = len(chamber_stats)
    if len(chamber_stats) > 0:
        bottleneck = chamber_stats.iloc[0]['CHAMBER_TYPE']
        bottleneck_time = chamber_stats.iloc[0]['Mean_Minutes']
    else:
        bottleneck = "N/A"
        bottleneck_time = 0
    
    summary = html.Div([
        html.Div([
            html.H3(f"📊 Analysis Summary for: {selected_recipe}", 
                   style={'color': '#2c3e50', 'margin-bottom': '15px'}),
            html.Div([
                html.Div([
                    html.H4(f"{total_wafers:,}", style={'color': '#e74c3c', 'margin': '0'}),
                    html.P("Total Wafers", style={'margin': '0', 'font-size': '14px'})
                ], style={'text-align': 'center', 'width': '20%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H4(f"{chamber_types}", style={'color': '#3498db', 'margin': '0'}),
                    html.P("Chamber Types", style={'margin': '0', 'font-size': '14px'})
                ], style={'text-align': 'center', 'width': '20%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H4(f"{bottleneck}", style={'color': '#e67e22', 'margin': '0'}),
                    html.P("Bottleneck", style={'margin': '0', 'font-size': '14px'})
                ], style={'text-align': 'center', 'width': '20%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H4(f"{bottleneck_time:.1f} min", style={'color': '#9b59b6', 'margin': '0'}),
                    html.P("Bottleneck Time", style={'margin': '0', 'font-size': '14px'})
                ], style={'text-align': 'center', 'width': '20%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H4(f"{len(selected_chambers)}/{len(chamber_stats)}", style={'color': '#27ae60', 'margin': '0'}),
                    html.P("Chambers Selected", style={'margin': '0', 'font-size': '14px'})
                ], style={'text-align': 'center', 'width': '20%', 'display': 'inline-block'})
            ])
        ])
    ], style={'padding': '20px', 'background-color': '#ecf0f1', 'border-radius': '10px', 'margin-bottom': '20px'})
    
    # 1. Performance Chart (Bar chart of average processing times)
    performance_fig = px.bar(
        chamber_stats.sort_values('Mean_Minutes', ascending=True),
        x='Mean_Minutes', 
        y='CHAMBER_TYPE',
        title='⏱️ Average Processing Time by Chamber Type',
        labels={'Mean_Minutes': 'Average Minutes per Wafer', 'CHAMBER_TYPE': 'Chamber Type'},
        color='Mean_Minutes',
        color_continuous_scale='RdYlBu_r'
    )
    performance_fig.update_layout(height=400, showlegend=False)
    
    # 2. Variability Chart (Bar chart of standard deviations)
    variability_fig = px.bar(
        chamber_stats.sort_values('Std_Minutes', ascending=False),
        x='CHAMBER_TYPE',
        y='Std_Minutes',
        title='📊 Processing Time Variation (Standard Deviation)',
        labels={'Std_Minutes': 'Standard Deviation (Minutes)', 'CHAMBER_TYPE': 'Chamber Type'},
        color='Std_Minutes',
        color_continuous_scale='Reds'
    )
    variability_fig.update_layout(height=400, showlegend=False)
    
    # 3. Utilization Chart (Bar chart of wafer counts)
    utilization_fig = px.bar(
        chamber_stats.sort_values('Count', ascending=False),
        x='CHAMBER_TYPE',
        y='Count',
        title='🔄 Chamber Type Utilization (Wafer Count)',
        labels={'Count': 'Number of Wafers Processed', 'CHAMBER_TYPE': 'Chamber Type'},
        color='Count',
        color_continuous_scale='Greens'
    )
    utilization_fig.update_layout(height=400, showlegend=False)
    
    # 4. Distribution Chart (Box plot)
    distribution_fig = go.Figure()
    
    for chamber_type in chamber_stats['CHAMBER_TYPE']:
        chamber_data = df[df['CHAMBER_TYPE'] == chamber_type]['PROCESSING_TIME_MINUTES']
        distribution_fig.add_trace(go.Box(
            y=chamber_data,
            name=chamber_type,
            boxmean='sd'
        ))
    
    distribution_fig.update_layout(
        title='📈 Processing Time Distribution by Chamber Type',
        yaxis_title='Processing Time (Minutes)',
        xaxis_title='Chamber Type',
        height=400
    )
    
    # 5. Individual Chamber Chart (Top 15 slowest chambers)
    individual_stats = df.groupby('CHAMBER').agg({
        'PROCESSING_TIME_MINUTES': ['count', 'mean', 'std']
    }).round(2)
    individual_stats.columns = ['Count', 'Mean_Minutes', 'Std_Minutes']
    individual_stats['CHAMBER_TYPE'] = individual_stats.index.str[:3]
    individual_stats = individual_stats.sort_values('Mean_Minutes', ascending=False).head(15)
    
    # Color code by standard deviation (higher std = more red)
    individual_fig = px.bar(
        individual_stats.reset_index(),
        x='Mean_Minutes',
        y='CHAMBER',
        orientation='h',
        title='🔧 Top 15 Slowest Individual Chambers',
        labels={'Mean_Minutes': 'Average Minutes per Wafer', 'CHAMBER': 'Chamber ID'},
        color='Std_Minutes',
        color_continuous_scale='RdYlGn_r'
    )
    individual_fig.update_layout(height=500)
    
    return summary, performance_fig, variability_fig, utilization_fig, distribution_fig, individual_fig

# Run the app
if __name__ == '__main__':
    print("Starting Track Chamber Analysis Dashboard...")
    print("Dashboard will be available at: http://127.0.0.1:8050")
    app.run(debug=True, port=8050, host='127.0.0.1')