import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class StaffingAnalysisDashboard:
    """
    A modular dashboard component for analyzing Staffing Ratios, ALOS, 
    and deviations from benchmarks.
    """

    def __init__(self, id_prefix="staff_kpi"):
        """
        Args:
            id_prefix (str): A unique string to prefix all component IDs. 
                             Essential for multi-tab apps to avoid ID conflicts.
        """
        self.id_prefix = id_prefix
        
        # Define Benchmarks
        self.BENCHMARKS = {
            'ICU': {'nurse': 2, 'doctor': 14, 'alos_min': 2, 'alos_max': 4},
            'surgery': {'nurse': 6, 'doctor': 15, 'alos_min': 1, 'alos_max': 8},
            'general_medicine': {'nurse': 6, 'doctor': 15, 'alos_min': 4, 'alos_max': 6},
            'emergency': {'nurse': 4, 'doctor': None, 'alos_min': 4/24, 'alos_max': 6/24}
        }
        
        # Load and Process Data immediately upon instantiation
        self._load_and_process_data()

    def _get_id(self, name):
        """Helper to generate prefixed IDs."""
        return f"{self.id_prefix}_{name}"

    def _load_and_process_data(self):
        """Internal method to load CSVs and perform feature engineering."""
        try:
            services_df = pd.read_csv('services_weekly.csv')
            staff_schedule_df = pd.read_csv('staff_schedule.csv')
            patients_df = pd.read_csv('patients.csv')
        except FileNotFoundError:
            # Fallback for demonstration if files aren't found
            print("Error: CSV files not found. Ensure 'services_weekly.csv', 'staff_schedule.csv', and 'patients.csv' exist.")
            return

        # Sanitize strings
        services_df['service'] = services_df['service'].str.strip()
        staff_schedule_df['service'] = staff_schedule_df['service'].str.strip()
        patients_df['service'] = patients_df['service'].str.strip()

        # --- PREPARE STAFF COUNTS (WEEKLY) ---
        present_staff = staff_schedule_df[staff_schedule_df['present'] == 1]
        staff_counts = present_staff.groupby(['week', 'service', 'role']).size().unstack(fill_value=0).reset_index()
        for role in ['doctor', 'nurse']:
            if role not in staff_counts.columns: staff_counts[role] = 0

        # --- PREPARE WEEKLY DATA ---
        merged_df = pd.merge(services_df[['week', 'month', 'service', 'patients_admitted']], staff_counts, on=['week', 'service'], how='left')
        merged_df['doctor'] = merged_df['doctor'].fillna(0)
        merged_df['nurse'] = merged_df['nurse'].fillna(0)

        # Fix Month and Quarter Calculation
        merged_df['month'] = (pd.Timestamp('2025-01-01') + pd.to_timedelta((merged_df['week'] - 1) * 7, unit='D')).dt.month
        merged_df['quarter'] = ((merged_df['month'] - 1) // 3) + 1
        merged_df['week_of_month'] = merged_df.groupby('month')['week'].rank(method='dense').astype(int)

        # ALOS Calculation
        patients_df['arrival_date'] = pd.to_datetime(patients_df['arrival_date'])
        patients_df['departure_date'] = pd.to_datetime(patients_df['departure_date'])
        patients_df['los'] = (patients_df['departure_date'] - patients_df['arrival_date']).dt.days
        patients_df['week'] = patients_df['arrival_date'].dt.isocalendar().week

        alos_weekly = patients_df.groupby(['week', 'service'])['los'].mean().reset_index(name='avg_los')
        merged_df = pd.merge(merged_df, alos_weekly, on=['week', 'service'], how='left')
        merged_df['avg_los'] = merged_df['avg_los'].fillna(0)

        self.available_services = sorted(merged_df['service'].unique())

        # Ratio Calculation Helper
        def calculate_ratios(df, role):
            col_name = f'{role}_ratio'
            def get_ratio(row):
                patients = row['patients_admitted']
                staff = row[role]
                if staff > 0: 
                    return patients / staff
                else:
                    return 0 
            df[col_name] = df.apply(get_ratio, axis=1)
            return df

        merged_df = calculate_ratios(merged_df, 'nurse')
        merged_df = calculate_ratios(merged_df, 'doctor')
        
        # Store weekly data in instance
        self.merged_df = merged_df

        # --- PREPARE DAILY DATA ---
        patients_df['day_name'] = patients_df['arrival_date'].dt.day_name()
        daily_patients = patients_df.groupby(['week', 'day_name', 'service']).size().reset_index(name='patients_admitted')
        daily_merged = pd.merge(daily_patients, staff_counts, on=['week', 'service'], how='left')
        daily_merged['doctor'] = daily_merged['doctor'].fillna(0)
        daily_merged['nurse'] = daily_merged['nurse'].fillna(0)

        daily_merged['month'] = (pd.Timestamp('2025-01-01') + pd.to_timedelta((daily_merged['week'] - 1) * 7, unit='D')).dt.month
        daily_merged['quarter'] = ((daily_merged['month'] - 1) // 3) + 1

        daily_merged = calculate_ratios(daily_merged, 'nurse')
        daily_merged = calculate_ratios(daily_merged, 'doctor')
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_merged['day_name'] = pd.Categorical(daily_merged['day_name'], categories=days_order, ordered=True)
        daily_merged = daily_merged.sort_values(['week', 'day_name'])
        
        # Store daily data in instance
        self.daily_merged = daily_merged

    def get_layout(self):
        """Returns the Dash layout for this specific dashboard instance."""
        return html.Div([
            html.H2("Staffing & Efficiency Analytics", style={'textAlign': 'center', 'fontFamily': 'Arial', 'marginTop': '10px'}),
            
            # CONTROLS
            html.Div([
                html.Div([
                    html.Label("1. Service:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id=self._get_id('service-dropdown'), options=[], value='surgery', clearable=False)
                ], style={'width': '22%'}),
                
                html.Div([
                    html.Label("2. Time Scale:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id=self._get_id('time-scale-dropdown'),
                        options=[{'label': 'Weekly', 'value': 'week'}, {'label': 'Monthly', 'value': 'month'}, {'label': 'Quarterly', 'value': 'quarter'}],
                        value='week', clearable=False
                    )
                ], style={'width': '22%'}),
                
                # View Type Container
                html.Div([
                    html.Label("3. View Type:", style={'fontWeight': 'bold', 'color': '#d62728'}),
                    dcc.Dropdown(
                        id=self._get_id('view-type-dropdown'),
                        options=[{'label': 'Line Chart', 'value': 'line'}, {'label': 'Heatmap', 'value': 'heatmap'}],
                        value='line', clearable=False
                    )
                ], id=self._get_id('view-type-container'), style={'width': '22%', 'display': 'none'}),
                
                html.Div([
                    html.Label("4. Role:", style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id=self._get_id('role-selector'),
                        options=[{'label': ' Nurse', 'value': 'nurse'}, {'label': ' Doctor', 'value': 'doctor'}],
                        value='nurse', inline=True
                    )
                ], style={'width': '28%', 'paddingTop': '10px'})
            ], style={'width': '95%', 'margin': '0 auto 10px auto', 'display': 'flex', 'justifyContent': 'space-between'}),

            # TIME INTERVAL SLIDER
            html.Div([
                html.Label("Select Time Interval:", style={'fontWeight': 'bold'}),
                dcc.RangeSlider(
                    id=self._get_id('time-range-slider'),
                    step=1,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '95%', 'margin': '0 auto 30px auto', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'}),

            # ROW 1: Staffing Ratio (Fade Container) + Deviation
            html.Div([
                # FADE Container for Staffing Ratio
                html.Div([
                    dbc.Fade(
                        dcc.Graph(
                            id=self._get_id('trend-chart'), 
                            config={'displayModeBar': False}, 
                            style={'height': '380px'},
                            clear_on_unhover=True
                        ),
                        id=self._get_id('fade-staff-line'),
                        is_in=True,
                        appear=False,
                        style={'transition': 'opacity 500ms ease-in-out', 'position': 'absolute', 'width': '100%', 'zIndex': '10'}
                    ),
                    dbc.Fade(
                        dcc.Graph(
                            id=self._get_id('heatmap-chart'), 
                            config={'displayModeBar': False}, 
                            style={'height': '380px'},
                            clear_on_unhover=True
                        ),
                        id=self._get_id('fade-staff-heat'),
                        is_in=False,
                        appear=False,
                        style={'transition': 'opacity 500ms ease-in-out', 'position': 'absolute', 'width': '100%', 'zIndex': '5'}
                    )
                ], style={'position': 'relative', 'height': '400px', 'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top', 'border': '1px solid #ccc', 'padding': '5px'}),
                
                # Deviation Chart
                html.Div([
                    dcc.Graph(
                        id=self._get_id('deviation-chart'), 
                        style={'height': '380px'},
                        clear_on_unhover=True
                    )
                ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top', 'float': 'right', 'border': '1px solid #ccc', 'padding': '5px', 'height': '400px'})
            ], style={'width': '95%', 'margin': '0 auto'}),

            # ROW 2: ALOS (Fade Container)
            html.Div([
                html.Div([
                    dbc.Fade(
                        dcc.Graph(
                            id=self._get_id('alos-line-chart'), 
                            config={'displayModeBar': False}, 
                            style={'height': '380px'},
                            clear_on_unhover=True
                        ),
                        id=self._get_id('fade-alos-line'),
                        is_in=True,
                        appear=False,
                        style={'transition': 'opacity 500ms ease-in-out', 'position': 'absolute', 'width': '100%', 'zIndex': '10'}
                    ),
                    dbc.Fade(
                        dcc.Graph(
                            id=self._get_id('alos-heatmap-chart'), 
                            config={'displayModeBar': False}, 
                            style={'height': '380px'},
                            clear_on_unhover=True
                        ),
                        id=self._get_id('fade-alos-heat'),
                        is_in=False,
                        appear=False,
                        style={'transition': 'opacity 500ms ease-in-out', 'position': 'absolute', 'width': '100%', 'zIndex': '5'}
                    )
                ], style={'position': 'relative', 'height': '380px', 'width': '100%'})
            ], style={'width': '95%', 'margin': '20px auto', 'border': '1px solid #ccc', 'padding': '10px', 'height': '400px'}),
        ])

    def register_callbacks(self, app):
        """
        Registers all callbacks for this dashboard instance with the main Dash app.
        """
        
        # 1. Update Service Options
        app.callback(
            [Output(self._get_id('service-dropdown'), 'options'),
             Output(self._get_id('service-dropdown'), 'value')],
            Input(self._get_id('role-selector'), 'value'),
            State(self._get_id('service-dropdown'), 'value')
        )(self._update_service_options)

        # 2. Update View Type Options and Visibility
        app.callback(
            [Output(self._get_id('view-type-dropdown'), 'options'),
             Output(self._get_id('view-type-dropdown'), 'value'),
             Output(self._get_id('view-type-container'), 'style')],
            [Input(self._get_id('time-scale-dropdown'), 'value')],
            [State(self._get_id('view-type-dropdown'), 'value')]
        )(self._update_view_options)

        # 3. Control Fade Visibility (View Transition Animation)
        app.callback(
            [Output(self._get_id('fade-staff-line'), 'is_in'),
             Output(self._get_id('fade-staff-heat'), 'is_in'),
             Output(self._get_id('fade-alos-line'), 'is_in'),
             Output(self._get_id('fade-alos-heat'), 'is_in')],
            [Input(self._get_id('view-type-dropdown'), 'value')]
        )(self._control_fade)

        # 4. Update Slider Range
        app.callback(
            [Output(self._get_id('time-range-slider'), 'min'),
             Output(self._get_id('time-range-slider'), 'max'),
             Output(self._get_id('time-range-slider'), 'value'),
             Output(self._get_id('time-range-slider'), 'marks')],
            Input(self._get_id('time-scale-dropdown'), 'value')
        )(self._update_slider_config)

        # 5. Update Charts (Master Callback)
        app.callback(
            [Output(self._get_id('trend-chart'), 'figure'),
             Output(self._get_id('heatmap-chart'), 'figure'),
             Output(self._get_id('alos-line-chart'), 'figure'),
             Output(self._get_id('alos-heatmap-chart'), 'figure'),
             Output(self._get_id('deviation-chart'), 'figure')],
            [Input(self._get_id('service-dropdown'), 'value'),
             Input(self._get_id('role-selector'), 'value'),
             Input(self._get_id('time-scale-dropdown'), 'value'),
             Input(self._get_id('time-range-slider'), 'value'),
             Input(self._get_id('trend-chart'), 'hoverData'),
             Input(self._get_id('heatmap-chart'), 'hoverData'),
             Input(self._get_id('alos-line-chart'), 'hoverData'),
             Input(self._get_id('alos-heatmap-chart'), 'hoverData'),
             Input(self._get_id('view-type-dropdown'), 'value')]
        )(self._update_charts)

    # --- INTERNAL CALLBACK LOGIC METHODS ---

    def _update_service_options(self, role, current_service):
        valid_services = [s for s in self.available_services if self.BENCHMARKS.get(s, {}).get(role) is not None]
        options = [{'label': s.replace('_', ' ').title(), 'value': s} for s in valid_services]
        new_value = current_service
        if current_service not in valid_services:
            new_value = valid_services[0] if valid_services else None
        return options, new_value

    def _update_view_options(self, time_scale, current_view):
        options_all = [{'label': 'Line Chart', 'value': 'line'}, {'label': 'Heatmap', 'value': 'heatmap'}]
        options_line = [{'label': 'Line Chart', 'value': 'line'}]
        
        if time_scale == 'week':
            return options_line, 'line', {'display': 'none'}
        else:
            return options_all, current_view, {'width': '22%', 'display': 'block'}

    def _control_fade(self, view_type):
        if view_type == 'heatmap':
            return False, True, False, True 
        return True, False, True, False

    def _update_slider_config(self, time_scale):
        if time_scale == 'week':
            min_v, max_v = 1, 52
        elif time_scale == 'month':
            min_v, max_v = 1, 12
        elif time_scale == 'quarter':
            min_v, max_v = 1, 4
        
        marks = {i: str(i) for i in range(min_v, max_v+1, 5)} if max_v > 20 else {i: str(i) for i in range(min_v, max_v+1)}
        return min_v, max_v, [min_v, max_v], marks

    def _update_charts(self, service, role, time_scale, slider_range, hover_line, hover_heat, hover_alos_line, hover_alos_heat, view_type):
        if not service:
            return [go.Figure().update_layout(title="Select Service")] * 5
            
        limits = self.BENCHMARKS.get(service, {})
        ratio_limit = limits.get(role)
        if ratio_limit is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Dynamic Animation Duration Logic
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        # We need to check if the trigger contained our ID prefix + a control name
        is_control_trigger = False
        for ctrl in ['service-dropdown', 'role-selector', 'time-scale-dropdown', 'time-range-slider']:
            if triggered_id == self._get_id(ctrl):
                is_control_trigger = True
                break
        
        anim_duration = 500 if is_control_trigger else 0

        # Data Filtering
        df = self.merged_df[self.merged_df['service'] == service].copy()
        
        if slider_range:
            start, end = slider_range
            if start == end:
                max_limit = 52 if time_scale == 'week' else (12 if time_scale == 'month' else 4)
                if end < max_limit: end += 1
                else: start = max(1, start - 1)
            
            if time_scale == 'week':
                df = df[(df['week'] >= start) & (df['week'] <= end)]
            elif time_scale == 'month':
                df = df[(df['month'] >= start) & (df['month'] <= end)]
            elif time_scale == 'quarter':
                df = df[(df['quarter'] >= start) & (df['quarter'] <= end)]
        
        # Aggregation
        if time_scale == 'week':
            plot_df = df
            x_col = 'week'
            time_label = "Week"
        else:
            plot_df = df.groupby(time_scale).agg({
                'patients_admitted': 'mean',
                'nurse': 'mean',
                'doctor': 'mean',
                'avg_los': 'mean'
            }).reset_index()
            # Recalculate ratios on aggregated means
            def get_ratio(row, r):
                return row['patients_admitted'] / row[r] if row[r] > 0 else 0
            
            plot_df[f'{role}_ratio'] = plot_df.apply(lambda row: get_ratio(row, role), axis=1)
            x_col = time_scale
            time_label = time_scale.title()
            
            if time_scale == 'month':
                plot_df['label'] = pd.to_datetime(plot_df['month'], format='%m').dt.month_name().str.slice(stop=3)
            elif time_scale == 'quarter':
                plot_df['label'] = 'Q' + plot_df['quarter'].astype(str)

        x_data = plot_df['label'] if 'label' in plot_df.columns else plot_df[x_col]
        
        alos_min, alos_max = limits.get('alos_min', 0), limits.get('alos_max', 10)
        y_col_ratio = f'{role}_ratio'
        title_role = role.title()

        # Hover Logic
        hover_index_staff = None
        
        # Check if interaction came from Staffing Charts
        is_staff_interaction = (triggered_id == self._get_id('trend-chart')) or (triggered_id == self._get_id('heatmap-chart'))
        if not triggered_id and (hover_line or (view_type == 'heatmap' and hover_heat)):
            is_staff_interaction = True

        active_hover = None
        if triggered_id == self._get_id('heatmap-chart'): active_hover = hover_heat
        elif triggered_id == self._get_id('trend-chart'): active_hover = hover_line
        elif not triggered_id: active_hover = hover_heat if (view_type=='heatmap') else hover_line
        
        if is_staff_interaction and active_hover:
            try:
                if 'points' in active_hover:
                    pt = active_hover['points'][0]
                    if 'x' in pt and isinstance(pt['x'], str):
                        hover_x_label = pt['x']
                        if 'label' in plot_df.columns:
                            indices = plot_df.index[plot_df['label'] == hover_x_label].tolist()
                            if indices: hover_index_staff = indices[0]
                    elif 'pointIndex' in pt:
                        hover_index_staff = pt['pointIndex']
            except: pass

        # 1. Staffing Ratio Chart
        fig_trend = go.Figure()
        valid_max = plot_df[y_col_ratio].max() if not plot_df.empty else 0
        if pd.isna(valid_max) or valid_max == 0: valid_max = ratio_limit
        y_range_max = max(valid_max, ratio_limit * 1.5)
        x_list = x_data.tolist() if hasattr(x_data, 'tolist') else list(x_data)
        
        hover_template_trend = f"{time_label}: %{{x}}<br>Avg. Load: %{{y:.2f}}<extra></extra>"
        
        fig_trend.add_trace(go.Scatter(x=x_list, y=[ratio_limit]*len(x_list), mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(228, 245, 255, 1.0)', name='Safe Zone', hoverinfo='skip', showlegend=False))
        fig_trend.add_trace(go.Scatter(x=x_list, y=[y_range_max*1.3]*len(x_list), mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 213, 213, 1.0)', name='Critical Zone', hoverinfo='skip', showlegend=False))
        fig_trend.add_trace(go.Scatter(x=x_data, y=plot_df[y_col_ratio], mode='lines+markers', name=f'{title_role} Load', line=dict(color='#2c3e50', width=3), marker=dict(color='#2c3e50', size=8), hovertemplate=hover_template_trend))
        
        if hover_index_staff is not None and hover_index_staff < len(x_data) and (time_scale == 'week' or view_type != 'heatmap'):
            fig_trend.add_vline(x=x_data.iloc[hover_index_staff] if hasattr(x_data, 'iloc') else list(x_data)[hover_index_staff], line_width=1, line_dash="dash", line_color="black")

        fig_trend.update_layout(title=f"<b>Average patients to {title_role} Ratio</b>", xaxis_title=time_scale.title(), yaxis_title=f"Patients per {title_role}", margin=dict(l=40, r=20, t=40, b=30), height=350, transition={'duration': anim_duration, 'easing': 'cubic-in-out'})

        # 2. Heatmap
        fig_heat = go.Figure()
        if time_scale != 'week':
            z_matrix = pd.DataFrame()
            x_title, y_title, hover_template_heat = "", "", ""
            
            # Use original filtered DF for heatmap data construction
            df_heat_source = self.merged_df[self.merged_df['service'] == service].copy()
            if slider_range: # Apply slider filter to source
                 # (Logic similar to above filter block)
                 pass 

            if time_scale == 'month':
                heat_df = df.groupby(['month', 'week_of_month']).agg({'patients_admitted':'sum', 'nurse':'mean', 'doctor':'mean'}).reset_index()
                def get_ratio_heat(row, r): return row['patients_admitted'] / row[r] if row[r] > 0 else 0
                heat_df[y_col_ratio] = heat_df.apply(lambda row: get_ratio_heat(row, role), axis=1)
                
                z_matrix = heat_df.pivot(index='week_of_month', columns='month', values=y_col_ratio)
                x_title, y_title = "Month", "Week of Month"
                z_matrix.columns = [pd.to_datetime(m, format='%m').month_name()[:3] for m in z_matrix.columns]
                hover_template_heat = "Month: %{x}<br>Week: %{y}<br>Avg. Load: %{z:.2f}<extra></extra>"
            elif time_scale == 'quarter':
                heat_df = df.groupby(['quarter', 'month']).agg({'patients_admitted':'mean', 'nurse':'mean', 'doctor':'mean'}).reset_index()
                def get_ratio_heat(row, r): return row['patients_admitted'] / row[r] if row[r] > 0 else 0
                heat_df[y_col_ratio] = heat_df.apply(lambda row: get_ratio_heat(row, role), axis=1)
                
                heat_df['month_name'] = pd.to_datetime(heat_df['month'], format='%m').dt.month_name().str.slice(stop=3)
                z_matrix = heat_df.pivot(index='month_name', columns='quarter', values=y_col_ratio)
                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                existing_months = [m for m in month_order if m in z_matrix.index]
                z_matrix = z_matrix.reindex(existing_months)
                z_matrix.columns = ['Q' + str(c) for c in z_matrix.columns]
                x_title, y_title = "Quarter", "Month"
                hover_template_heat = "Quarter: %{x}<br>Month: %{y}<br>Avg. Load: %{z:.2f}<extra></extra>"

            if not z_matrix.empty:
                fig_heat.add_trace(go.Heatmap(z=z_matrix.values, x=z_matrix.columns, y=z_matrix.index, colorscale='RdBu_r', zmid=ratio_limit, xgap=3, ygap=3, hovertemplate=hover_template_heat))
                fig_heat.update_layout(title=f"<b>Average patients to {title_role} Ratio</b>", xaxis_title=x_title, yaxis_title=y_title, margin=dict(l=40, r=20, t=40, b=30), height=350, transition={'duration': anim_duration, 'easing': 'cubic-in-out'})

        # 3. ALOS Line
        base_colors = ['#1f77b4' if (alos_min <= x <= alos_max) else '#d62728' for x in plot_df['avg_los']]
        if hover_index_staff is not None and hover_index_staff < len(plot_df):
            final_colors, opacities = ['lightgrey']*len(plot_df), [0.3]*len(plot_df)
            final_colors[hover_index_staff], opacities[hover_index_staff] = base_colors[hover_index_staff], 1.0
        else:
            final_colors, opacities = base_colors, [1.0]*len(plot_df)

        fig_alos = go.Figure()
        hover_template_alos = f"{time_label}: %{{x}}<br>ALOS: %{{y:.2f}}<extra></extra>"
        
        target_text = "Target: 4-6 Hours" if service == 'emergency' else f"Target: {alos_min}-{alos_max} Days"
        fig_alos.add_hrect(y0=alos_min, y1=alos_max, fillcolor="#1f77b4", opacity=0.1, layer="below", line_width=0, annotation_text=target_text)
        fig_alos.add_trace(go.Scatter(x=x_data, y=plot_df['avg_los'], mode='lines', line=dict(color='lightgrey'), hoverinfo='skip'))
        fig_alos.add_trace(go.Scatter(x=x_data, y=plot_df['avg_los'], mode='markers', marker=dict(color=final_colors, size=12, opacity=opacities), hovertemplate=hover_template_alos))
        
        if hover_index_staff is not None and hover_index_staff < len(x_data) and (time_scale == 'week' or view_type != 'heatmap'):
            fig_alos.add_vline(x=x_data.iloc[hover_index_staff] if hasattr(x_data, 'iloc') else list(x_data)[hover_index_staff], line_width=1, line_dash="dash", line_color="black")

        fig_alos.update_layout(title="<b>Patients Average Length of Stay</b>", xaxis_title=time_scale.title(), yaxis_title="Days", height=350, margin=dict(l=40, r=20, t=40, b=30), showlegend=False, transition={'duration': anim_duration, 'easing': 'cubic-in-out'})

        # 4. ALOS Heatmap (Simplified for brevity, mirrors structure of Staff Heatmap)
        fig_alos_heat = go.Figure()
        if time_scale != 'week':
            # (Reusable logic for ALOS heatmap matrix construction would go here)
            # For strict refactoring, reusing the pivot logic from Heatmap section but with 'avg_los' value
            if time_scale == 'month':
                 heat_df_alos = df.groupby(['month', 'week_of_month'])['avg_los'].mean().reset_index()
                 z_matrix_alos = heat_df_alos.pivot(index='week_of_month', columns='month', values='avg_los')
                 z_matrix_alos.columns = [pd.to_datetime(m, format='%m').month_name()[:3] for m in z_matrix_alos.columns]
                 x_title_alos, y_title_alos = "Month", "Week of Month"
            elif time_scale == 'quarter':
                 heat_df_alos = df.groupby(['quarter', 'month'])['avg_los'].mean().reset_index()
                 heat_df_alos['month_name'] = pd.to_datetime(heat_df_alos['month'], format='%m').dt.month_name().str.slice(stop=3)
                 z_matrix_alos = heat_df_alos.pivot(index='month_name', columns='quarter', values='avg_los')
                 month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                 z_matrix_alos = z_matrix_alos.reindex([m for m in month_order if m in z_matrix_alos.index])
                 z_matrix_alos.columns = ['Q' + str(c) for c in z_matrix_alos.columns]
                 x_title_alos, y_title_alos = "Quarter", "Month"

            if not z_matrix_alos.empty:
                fig_alos_heat.add_trace(go.Heatmap(z=z_matrix_alos.values, x=z_matrix_alos.columns, y=z_matrix_alos.index, colorscale='RdBu_r', zmid=alos_max, xgap=3, ygap=3))
                fig_alos_heat.update_layout(title="<b>Patients Average Length of Stay</b>", xaxis_title=x_title_alos, yaxis_title=y_title_alos, margin=dict(l=40, r=20, t=40, b=30), height=350, transition={'duration': anim_duration, 'easing': 'cubic-in-out'})

        # 5. Deviation
        plot_df['dev'] = (plot_df[y_col_ratio] - ratio_limit)
        dev_colors = ['#d62728' if x > 0 else '#1f77b4' for x in plot_df['dev']]
        
        if hover_index_staff is not None:
            bar_opacities = [1.0 if i == hover_index_staff else 0.3 for i in range(len(plot_df))]
        else:
            bar_opacities = [1.0] * len(plot_df)

        fig_dev = go.Figure(go.Bar(x=x_data, y=plot_df['dev'], marker_color=dev_colors, marker_opacity=bar_opacities))
        fig_dev.update_layout(title="<b>Staffing Deviation from Benchmark</b>", height=350, margin=dict(l=40, r=20, t=40, b=30), transition={'duration': anim_duration, 'easing': 'cubic-in-out'})

        return fig_trend, fig_heat, fig_alos, fig_alos_heat, fig_dev