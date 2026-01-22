import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from collections import Counter

# ==========================================
# 1. HELPERS & LOGIC (Untouched Preprocessing)
# ==========================================
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by='week')
    
    # Robust Cleaning
    df['event'] = df['event'].astype(str).str.strip().str.lower()
    df['event'] = df['event'].replace({'nan': 'none', '': 'none'})
    df['month'] = (pd.Timestamp('2025-01-01') + pd.to_timedelta((df['week'] - 1) * 7, unit='D')).dt.month
    
    # Pre-calculate quarters
    df['quarter'] = ((df['month'] - 1) // 3) + 1
    return df
    
class HospitalDemandChart:
    def __init__(self, dataframe, colors):
        self.df = dataframe
        self.colors = colors

    def _aggregate_data(self, dff, timescale):
        def join_events(series):
            events = [e for e in series if e != 'none']
            return ",".join(events) if events else 'none'

        if timescale == 'week':
            plot_df = dff.sort_values('week')
            x_col = 'week'
            x_title = "Week Number"
            marker_size = 12
        else:
            group_col = 'month' if timescale == 'month' else 'quarter'
            x_title = timescale.title()
            marker_size = 15 if timescale == 'month' else 18
            
            plot_df = dff.groupby(group_col).agg({
                'patients_request': 'sum',
                'available_beds': 'sum',
                'event': join_events 
            }).reset_index()
            x_col = group_col
            
        return plot_df, x_col, x_title, marker_size

    def _process_events(self, plot_df, selected_events, timescale):
        def filter_and_format(event_str):
            if event_str == 'none' or not selected_events:
                return 'none', 0
            
            all_events = event_str.split(',')
            filtered = [e for e in all_events if e in selected_events]
            
            if not filtered:
                return 'none', 0
            
            counts = Counter(filtered)
            formatted_list = []
            for event_name, count in sorted(counts.items()):
                label = f"{count}x {event_name.title()}" if count > 1 else event_name.title()
                formatted_list.append(label)
            
            return "<br>".join(formatted_list), len(filtered)

        temp_results = plot_df['event'].apply(filter_and_format)
        plot_df['display_event'] = [res[0] for res in temp_results]
        plot_df['event_count'] = [res[1] for res in temp_results]
        return plot_df

    def create_figure(self, service, timescale, selected_events):
        dff = self.df[self.df['service'] == service].copy()
        if dff.empty:
            return go.Figure()

        plot_df, x_col, x_title, marker_size = self._aggregate_data(dff, timescale)
        plot_df = self._process_events(plot_df, selected_events, timescale)

        y_demand = plot_df['patients_request']
        y_capacity = plot_df['available_beds']
        plot_df['deficit'] = y_demand - y_capacity
        
        max_deficit_row = None
        if not plot_df.empty:
            max_idx = plot_df['deficit'].idxmax()
            if plot_df.loc[max_idx, 'deficit'] > 0:
                max_deficit_row = plot_df.loc[max_idx]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=plot_df[x_col], y=y_capacity, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=plot_df[x_col], y=np.maximum(y_demand, y_capacity),
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=self.colors['deficit'],
            name='Deficit', hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(x=plot_df[x_col], y=y_capacity, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=plot_df[x_col], y=np.minimum(y_demand, y_capacity),
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=self.colors['surplus'],
            name='Surplus', hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=plot_df[x_col], y=y_capacity,
            mode='lines', name='Capacity',
            line=dict(color=self.colors['capacity'], width=2, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=plot_df[x_col], y=y_demand,
            mode='lines+markers', name='Demand',
            line=dict(color=self.colors['demand'], width=3),
            marker=dict(size=6)
        ))

        event_rows = plot_df[plot_df['event_count'] > 0]
        if not event_rows.empty:
            fig.add_trace(go.Scatter(
                x=event_rows[x_col], y=event_rows['patients_request'],
                mode='markers+text', name='Event',
                marker=dict(symbol='diamond', size=marker_size, color=self.colors['event'], line=dict(color='black', width=1)),
                text=event_rows['event_count'] if timescale != 'week' else "",
                textposition="top center",
                textfont=dict(color=self.colors['text'], size=12),
                customdata=np.stack((event_rows[x_col], event_rows['display_event']), axis=-1),
                hovertemplate=f"<b>{x_title} %{{customdata[0]}}</b><br>Events Summary:<br>%{{customdata[1]}}<extra></extra>"
            ))

        if max_deficit_row is not None:
            fig.add_annotation(
                x=max_deficit_row[x_col], y=max_deficit_row['patients_request'],
                text=f"<b>Peak Deficit</b><br>{int(max_deficit_row['deficit'])}",
                showarrow=True, arrowhead=2, arrowcolor=self.colors['demand'],
                ax=0, ay=-40, bgcolor="white", bordercolor=self.colors['demand']
            )

        fig.update_layout(
            title=dict(text=f"<b>{service.title()}</b>: Demand vs. Capacity ({timescale.title()} View)", font=dict(size=18, color=self.colors['text'])),
            xaxis=dict(title=x_title, showgrid=False, linecolor=self.colors['grid'], rangeslider=dict(visible=True, thickness=0.1, bgcolor=self.colors['bg']), type="linear"),
            yaxis=dict(title="Total Requests" if timescale != 'week' else "Requests", showgrid=True, gridcolor='#f0f0f0'),
            template="plotly_white", hovermode="x",
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

# ==========================================
# 2. DASHBOARD COMPONENT CLASS
# ==========================================
class DemandAnalysisDashboard:
    def __init__(self, csv_path, id_prefix=""):
        """
        Initializes the dashboard component.
        :param csv_path: Path to the CSV file.
        :param id_prefix: Unique string to prefix HTML IDs (essential if using multiple instances).
        """
        self.id_prefix = id_prefix
        
        # 1. Load Data
        self.df = load_data(csv_path)
        
        # 2. Config
        self.COLORS = {
            'bg': '#f4f6f8', 'card': '#ffffff', 'text': '#2c3e50',
            'demand': '#e74c3c', 'capacity': "#2680bc",
            'deficit': 'rgba(231, 76, 60, 0.4)', 
            'surplus': 'rgba(38, 128, 188, 0.3)',
            'grid': '#bdc3c7', 'event': '#f1c40f'
        }
        
        # 3. Initialize Logic Engine
        self.chart_generator = HospitalDemandChart(self.df, self.COLORS)

    def layout(self):
        """Returns the Dash HTML layout for this component."""
        return html.Div(style={'backgroundColor': self.COLORS['bg'], 'fontFamily': 'Roboto, sans-serif', 'padding': '20px'}, children=[
            
            # HEADER
            html.Div([
                html.H2("Hospital Demand Analysis", style={'color': self.COLORS['text'], 'fontWeight': '700', 'marginBottom': '5px'})
            ], style={'marginBottom': '20px', 'textAlign': 'center'}),

            # MAIN CARD
            html.Div(style={'backgroundColor': self.COLORS['card'], 'padding': '20px', 'borderRadius': '12px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.05)'}, children=[
                
                # CONTROLS
                html.Div([
                    # TIMESCALE
                    html.Div([
                        html.Label("TIMESCALE VIEW", style={'fontWeight': 'bold', 'color': '#95a5a6', 'fontSize': '11px'}),
                        dcc.Dropdown(
                            id=f'{self.id_prefix}timescale-selector',
                            options=[
                                {'label': 'Weekly', 'value': 'week'},
                                {'label': 'Monthly', 'value': 'month'},
                                {'label': 'Quarterly', 'value': 'quarter'}
                            ],
                            value='week', clearable=False
                        )
                    ], style={'width': '200px', 'display': 'inline-block', 'marginRight': '20px'}),

                    # SERVICE
                    html.Div([
                        html.Label("SERVICE", style={'fontWeight': 'bold', 'color': '#95a5a6', 'fontSize': '11px'}),
                        dcc.Dropdown(
                            id=f'{self.id_prefix}service-selector',
                            options=[{'label': s.replace('_', ' ').title(), 'value': s} for s in sorted(self.df['service'].unique())],
                            value='emergency', clearable=False
                        )
                    ], style={'width': '180px', 'display': 'inline-block', 'marginRight': '20px'}),

                    # FILTER EVENTS
                    html.Div([
                        html.Label("FILTER EVENTS", style={'fontWeight': 'bold', 'color': '#95a5a6', 'fontSize': '11px'}),
                        dcc.Dropdown(
                            id=f'{self.id_prefix}event-selector',
                            options=[
                                {'label': 'Strike', 'value': 'strike'},
                                {'label': 'Donation', 'value': 'donation'},
                                {'label': 'Flu', 'value': 'flu'}
                            ],
                            value=['strike', 'donation', 'flu'], 
                            multi=True,
                            placeholder="Select events..."
                        )
                    ], style={'width': '300px', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Span("ℹ️ Drag handles to zoom.", style={'color': '#95a5a6', 'fontSize': '12px'})
                    ], style={'float': 'right', 'paddingTop': '25px'})

                ], style={'marginBottom': '20px', 'borderBottom': '1px solid #f0f0f0', 'paddingBottom': '20px'}),

                # CHART CONTAINER
                dcc.Graph(id=f'{self.id_prefix}trend-chart', config={'displayModeBar': False})
            ])
        ])

    def register_callbacks(self, app):
        """Registers the specific callbacks for this instance with the main app."""
        @app.callback(
            Output(f'{self.id_prefix}trend-chart', 'figure'),
            [Input(f'{self.id_prefix}service-selector', 'value'), 
             Input(f'{self.id_prefix}timescale-selector', 'value'),
             Input(f'{self.id_prefix}event-selector', 'value')]
        )
        def update_chart(service, timescale, selected_events):
            return self.chart_generator.create_figure(service, timescale, selected_events)

# ==========================================
# 3. APP SETUP & EXECUTION
# ==========================================
if __name__ == '__main__':
    external_stylesheets = ['https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    # Instantiate the dashboard class
    # You can now easily create multiple instances with different IDs
    dashboard1 = DemandAnalysisDashboard("services_weekly.csv", id_prefix="tab1-")
    
    # Setup App Layout
    app.layout = html.Div([
        dashboard1.layout()
    ])

    # Register Callbacks
    dashboard1.register_callbacks(app)

    app.run_server()
