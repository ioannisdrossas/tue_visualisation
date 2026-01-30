import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from jbi100_app.services.staffing_service import StaffingDataManager

class StaffingAnalysisDashboard:
    """
    Class for staffing analytics 
    (separate tab in the app script)
    """

    def __init__(self, services_csv:str, schedule_csv:str, patients_csv:str, id_prefix="staff_kpi"):
        """
            Initializes the dashboard instance and the internal data manager.

            Args:
                services_csv (str): Path to the services CSV file.
                schedule_csv (str): Path to the schedule CSV file.
                patients_csv (str): Path to the patients CSV file.
                id_prefix (str): Unique identifier prefix for dashboard components.

            Returns:
                None
        """
        self.id_prefix = id_prefix
        
        # Instantiate the Data Manager
        self.data_manager = StaffingDataManager(services_csv, schedule_csv, patients_csv)

    def _get_id(self, name:str) -> str:
        """
            Generates a unique component ID using the class prefix.

            Args:
                name (str): The base name of the component.

            Returns:
                str: The full unique ID string.
        """
        return f"{self.id_prefix}_{name}"

    def get_layout(self) -> html.Div:
        """
            Constructs and returns the HTML layout for the dashboard tab.

            Args:
                None

            Returns:
                html.Div: The root Dash HTML component containing the dashboard layout.
        """
        return html.Div([
            
            # CONTROLS
            html.Div([
                html.Div([
                    html.Label("Service:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id=self._get_id('service-dropdown'), options=[], value='surgery', clearable=False)
                ], style={'width': '22%'}),
                
                html.Div([
                    html.Label("Time Scale:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id=self._get_id('time-scale-dropdown'),
                        options=[{'label': 'Weekly', 'value': 'week'}, {'label': 'Monthly', 'value': 'month'}, {'label': 'Quarterly', 'value': 'quarter'}],
                        value='week', clearable=False
                    )
                ], style={'width': '22%'}),
                
                # View Type Container
                html.Div([
                    html.Label("View Type:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id=self._get_id('view-type-dropdown'),
                        options=[{'label': 'Line Chart', 'value': 'line'}, {'label': 'Heatmap', 'value': 'heatmap'}],
                        value='line', clearable=False
                    )
                ], id=self._get_id('view-type-container'), style={'width': '22%', 'display': 'none'}),
                
                html.Div([
                    html.Label("Role:", style={'fontWeight': 'bold'}),
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

    def register_callbacks(self, app: dash.Dash):
        """
            Registers all Dash callbacks required for this dashboard instance.

            Args:
                app (dash.Dash): The main Dash application instance.
        """
        
        # Update Service Options
        app.callback(
            [Output(self._get_id('service-dropdown'), 'options'),
             Output(self._get_id('service-dropdown'), 'value')],
            Input(self._get_id('role-selector'), 'value'),
            State(self._get_id('service-dropdown'), 'value')
        )(self._update_service_options)

        # Update View Type Options and Visibility
        app.callback(
            [Output(self._get_id('view-type-dropdown'), 'options'),
             Output(self._get_id('view-type-dropdown'), 'value'),
             Output(self._get_id('view-type-container'), 'style')],
            [Input(self._get_id('time-scale-dropdown'), 'value')],
            [State(self._get_id('view-type-dropdown'), 'value')]
        )(self._update_view_options)

        # Control Fade Visibility (View Transition Animation)
        app.callback(
            [Output(self._get_id('fade-staff-line'), 'is_in'),
             Output(self._get_id('fade-staff-heat'), 'is_in'),
             Output(self._get_id('fade-alos-line'), 'is_in'),
             Output(self._get_id('fade-alos-heat'), 'is_in')],
            [Input(self._get_id('view-type-dropdown'), 'value')]
        )(self._control_fade)

        # Update Slider Range
        app.callback(
            [Output(self._get_id('time-range-slider'), 'min'),
             Output(self._get_id('time-range-slider'), 'max'),
             Output(self._get_id('time-range-slider'), 'value'),
             Output(self._get_id('time-range-slider'), 'marks')],
            Input(self._get_id('time-scale-dropdown'), 'value')
        )(self._update_slider_config)

        # Update Charts (Master Callback)
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

    # Internal Callback Methods

    def _update_service_options(self, role: str, current_service: str) -> tuple[list[dict], str | None]:
        """
            Updates the service dropdown options based on the selected staff role.

            Args:
                role (str): The selected staff role (e.g., 'nurse' or 'doctor').
                current_service (str): The currently selected service value.

            Returns:
                tuple[list[dict], str | None]: A tuple containing the list of dropdown options and the new selected value.
        """
        valid_services = [s for s in self.data_manager.available_services if self.data_manager.get_benchmarks(s).get(role) is not None]
        options = [{'label': s.replace('_', ' ').title(), 'value': s} for s in valid_services]
        new_value = current_service
        if current_service not in valid_services:
            new_value = valid_services[0] if valid_services else None
        return options, new_value

    def _update_view_options(self, time_scale: str, current_view: str) -> tuple[list[dict], str, dict]:
        """
            Updates the view type dropdown (Line vs Heatmap) based on the time scale.

            Args:
                time_scale (str): The selected time scale ('week', 'month', or 'quarter').
                current_view (str): The currently selected view type.

            Returns:
                tuple[list[dict], str, dict]: A tuple containing options, the selected value, and the container style dict.
        """
        options_all = [{'label': 'Line Chart', 'value': 'line'}, {'label': 'Heatmap', 'value': 'heatmap'}]
        options_line = [{'label': 'Line Chart', 'value': 'line'}]
        
        if time_scale == 'week':
            return options_line, 'line', {'display': 'none'}
        else:
            return options_all, current_view, {'width': '22%', 'display': 'block'}

    def _control_fade(self, view_type: str) -> tuple[bool, bool, bool, bool]:
        """
            Controls the visibility (fade) of the line charts vs heatmaps.

            Args:
                view_type (str): The selected view type ('line' or 'heatmap').

            Returns:
                tuple[bool, bool, bool, bool]: A tuple of booleans indicating which components are 'in' (visible).
        """
        if view_type == 'heatmap':
            return False, True, False, True 
        return True, False, True, False

    def _update_slider_config(self, time_scale: str) -> tuple[int, int, list[int], dict]:
        """
            Updates the range slider configuration based on the selected time scale.

            Args:
                time_scale (str): The selected time scale ('week', 'month', or 'quarter').

            Returns:
                tuple[int, int, list[int], dict]: A tuple containing min value, max value, default value list, and marks dict.
        """
        if time_scale == 'week':
            min_v, max_v = 1, 52
        elif time_scale == 'month':
            min_v, max_v = 1, 12
        elif time_scale == 'quarter':
            min_v, max_v = 1, 4
        
        marks = {i: str(i) for i in range(min_v, max_v+1, 5)} if max_v > 20 else {i: str(i) for i in range(min_v, max_v+1)}
        return min_v, max_v, [min_v, max_v], marks

    def _update_charts(self, service, role, time_scale, slider_range, 
                       hover_line, hover_heat, hover_alos_line, 
                       hover_alos_heat, view_type) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure]:
        """
            Master callback to update all charts based on user inputs.

            Args:
                service (str): Selected service name.
                role (str): Selected staff role.
                time_scale (str): Selected time scale.
                slider_range (list[int]): Start and end values from the slider.
                hover_line (dict): Hover data from the trend line chart.
                hover_heat (dict): Hover data from the staffing heatmap.
                hover_alos_line (dict): Hover data from the ALOS line chart.
                hover_alos_heat (dict): Hover data from the ALOS heatmap.
                view_type (str): Selected view type ('line' or 'heatmap').

            Returns:
                tuple[go.Figure, ...]: A tuple containing 5 Plotly Figure objects.
        """
        if not service:
            return [go.Figure().update_layout(title="Select Service")] * 5
            
        limits = self.data_manager.get_benchmarks(service)
        ratio_limit = limits.get(role)
        if ratio_limit is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Dynamic Animation Duration Logic
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        is_control_trigger = False
        for ctrl in ['service-dropdown', 'role-selector', 'time-scale-dropdown', 'time-range-slider']:
            if triggered_id == self._get_id(ctrl):
                is_control_trigger = True
                break
        anim_duration = 500 if is_control_trigger else 0

        # Data Filtering
        df = self.data_manager.merged_df[self.data_manager.merged_df['service'] == service].copy()
        
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
        
        # Determine Hover Index
        hover_index_staff = None
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

        # Build individual figures using new helper methods
        fig_trend = self._build_trend_fig(plot_df, x_data, role, limits, ratio_limit, time_scale, time_label, view_type, hover_index_staff, anim_duration)
        fig_heat = self._build_heatmap_fig(service, role, time_scale, ratio_limit, slider_range, anim_duration)
        fig_alos = self._build_alos_fig(plot_df, x_data, service, limits, time_scale, time_label, view_type, hover_index_staff, anim_duration)
        fig_alos_heat = self._build_alos_heatmap_fig(service, time_scale, limits, slider_range, anim_duration)
        fig_dev = self._build_deviation_fig(plot_df, x_data, role, ratio_limit, hover_index_staff, anim_duration)

        return fig_trend, fig_heat, fig_alos, fig_alos_heat, fig_dev

    # --- New Split Methods ---

    def _build_trend_fig(self, plot_df: pd.DataFrame, x_data: list | pd.Series, role: str, 
                         limits: dict, ratio_limit: float, time_scale: str, 
                         time_label: str, view_type: str, hover_index_staff: int | None, 
                         anim_duration: int) -> go.Figure:
        """
            Builds the Staffing Ratio Trend Line Chart.

            Args:
                plot_df (pd.DataFrame): The filtered data for plotting.
                x_data (list | pd.Series): The x-axis data (labels or numbers).
                role (str): The selected staff role.
                limits (dict): Benchmark limits for the service.
                ratio_limit (float): The specific ratio limit for the role.
                time_scale (str): The current time scale.
                time_label (str): Label for the time axis.
                view_type (str): The current view type.
                hover_index_staff (int | None): The index of the currently hovered data point.
                anim_duration (int): Duration for chart transitions in ms.

            Returns:
                go.Figure: The configured Plotly figure for the trend chart.
        """
        fig_trend = go.Figure()
        y_col_ratio = f'{role}_ratio'
        title_role = role.title()
        
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
        return fig_trend

    def _build_heatmap_fig(self, service: str, role: str, time_scale: str, ratio_limit: float, slider_range: list[int], anim_duration: int) -> go.Figure:
        """
            Builds the Staffing Ratio Heatmap.

            Args:
                service (str): The selected service.
                role (str): The selected staff role.
                time_scale (str): The current time scale.
                ratio_limit (float): The benchmark ratio limit.
                slider_range (list[int]): The time range filter values.
                anim_duration (int): Duration for chart transitions in ms.

            Returns:
                go.Figure: The configured Plotly figure for the heatmap.
        """
        fig_heat = go.Figure()
        y_col_ratio = f'{role}_ratio'
        title_role = role.title()
        
        if time_scale == 'week':
            return fig_heat # Heatmap not applicable for weekly view
            
        z_matrix = pd.DataFrame()
        x_title, y_title, hover_template_heat = "", "", ""
        
        # Use original filtered DF from data manager for heatmap construction
        df = self.data_manager.merged_df[self.data_manager.merged_df['service'] == service].copy()
        if slider_range:
            start, end = slider_range
            if start == end:
                max_limit = 12 if time_scale == 'month' else 4
                if end < max_limit: end += 1
                else: start = max(1, start - 1)
            
            if time_scale == 'month':
                df = df[(df['month'] >= start) & (df['month'] <= end)]
            elif time_scale == 'quarter':
                df = df[(df['quarter'] >= start) & (df['quarter'] <= end)]

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
        
        return fig_heat

    def _build_alos_fig(self, plot_df: pd.DataFrame, x_data: list | pd.Series, service: str, 
                        limits: dict, time_scale: str, time_label: str, 
                        view_type: str, hover_index_staff: int | None, anim_duration: int) -> go.Figure:
        """
            Builds the Average Length of Stay (ALOS) Line Chart.

            Args:
                plot_df (pd.DataFrame): The filtered data for plotting.
                x_data (list | pd.Series): The x-axis data.
                service (str): The selected service.
                limits (dict): Benchmark limits (min/max ALOS).
                time_scale (str): The current time scale.
                time_label (str): Label for the time axis.
                view_type (str): The current view type.
                hover_index_staff (int | None): The index of the currently hovered data point.
                anim_duration (int): Duration for chart transitions in ms.

            Returns:
                go.Figure: The configured Plotly figure for the ALOS chart.
        """
        fig_alos = go.Figure()
        alos_min, alos_max = limits.get('alos_min', 0), limits.get('alos_max', 10)
        
        base_colors = ['#1f77b4' if (alos_min <= x <= alos_max) else '#d62728' for x in plot_df['avg_los']]
        if hover_index_staff is not None and hover_index_staff < len(plot_df):
            final_colors, opacities = ['lightgrey']*len(plot_df), [0.3]*len(plot_df)
            final_colors[hover_index_staff], opacities[hover_index_staff] = base_colors[hover_index_staff], 1.0
        else:
            final_colors, opacities = base_colors, [1.0]*len(plot_df)

        hover_template_alos = f"{time_label}: %{{x}}<br>ALOS: %{{y:.2f}}<extra></extra>"
        
        target_text = "Target: 4-6 Hours" if service == 'emergency' else f"Target: {alos_min}-{alos_max} Days"
        fig_alos.add_hrect(y0=alos_min, y1=alos_max, fillcolor="#1f77b4", opacity=0.1, layer="below", line_width=0, annotation_text=target_text)
        fig_alos.add_trace(go.Scatter(x=x_data, y=plot_df['avg_los'], mode='lines', line=dict(color='lightgrey'), hoverinfo='skip'))
        fig_alos.add_trace(go.Scatter(x=x_data, y=plot_df['avg_los'], mode='markers', marker=dict(color=final_colors, size=12, opacity=opacities), hovertemplate=hover_template_alos))
        
        if hover_index_staff is not None and hover_index_staff < len(x_data) and (time_scale == 'week' or view_type != 'heatmap'):
            fig_alos.add_vline(x=x_data.iloc[hover_index_staff] if hasattr(x_data, 'iloc') else list(x_data)[hover_index_staff], line_width=1, line_dash="dash", line_color="black")

        fig_alos.update_layout(title="<b>Patients Average Length of Stay</b>", xaxis_title=time_scale.title(), yaxis_title="Days", height=350, margin=dict(l=40, r=20, t=40, b=30), showlegend=False, transition={'duration': anim_duration, 'easing': 'cubic-in-out'})
        return fig_alos

    def _build_alos_heatmap_fig(self, service: str, time_scale: str, limits: dict, 
                                slider_range: list[int], anim_duration: int) -> go.Figure:
        """
            Builds the Average Length of Stay (ALOS) Heatmap.

            Args:
                service (str): The selected service.
                time_scale (str): The current time scale.
                limits (dict): Benchmark limits (min/max ALOS).
                slider_range (list[int]): The time range filter values.
                anim_duration (int): Duration for chart transitions in ms.

            Returns:
                go.Figure: The configured Plotly figure for the ALOS heatmap.
        """
        fig_alos_heat = go.Figure()
        alos_max = limits.get('alos_max', 10)
        
        if time_scale == 'week':
            return fig_alos_heat
            
        z_matrix_alos = pd.DataFrame()
        x_title_alos, y_title_alos = "", ""

        # Use original filtered DF from data manager
        df = self.data_manager.merged_df[self.data_manager.merged_df['service'] == service].copy()
        if slider_range:
            start, end = slider_range
            if start == end:
                max_limit = 12 if time_scale == 'month' else 4
                if end < max_limit: end += 1
                else: start = max(1, start - 1)
            
            if time_scale == 'month':
                df = df[(df['month'] >= start) & (df['month'] <= end)]
            elif time_scale == 'quarter':
                df = df[(df['quarter'] >= start) & (df['quarter'] <= end)]

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
        
        return fig_alos_heat

    def _build_deviation_fig(self, plot_df: pd.DataFrame, x_data: list | pd.Series, role: str, 
                             ratio_limit: float, hover_index_staff: int | None, anim_duration: int) -> go.Figure:
        """
            Builds the Staffing Deviation Bar Chart.

            Args:
                plot_df (pd.DataFrame): The filtered data for plotting.
                x_data (list | pd.Series): The x-axis data.
                role (str): The selected staff role.
                ratio_limit (float): The benchmark ratio limit to calculate deviation from.
                hover_index_staff (int | None): The index of the currently hovered data point.
                anim_duration (int): Duration for chart transitions in ms.

            Returns:
                go.Figure: The configured Plotly figure for the deviation chart.
        """
        y_col_ratio = f'{role}_ratio'
        
        dev_values = (plot_df[y_col_ratio] - ratio_limit)
        dev_colors = ['#d62728' if x > 0 else '#1f77b4' for x in dev_values]
        
        if hover_index_staff is not None:
            bar_opacities = [1.0 if i == hover_index_staff else 0.3 for i in range(len(plot_df))]
        else:
            bar_opacities = [1.0] * len(plot_df)

        fig_dev = go.Figure(go.Bar(x=x_data, y=dev_values, marker_color=dev_colors, marker_opacity=bar_opacities))
        fig_dev.update_layout(title="<b>Staffing Deviation from Benchmark</b>", height=350, margin=dict(l=40, r=20, t=40, b=30), transition={'duration': anim_duration, 'easing': 'cubic-in-out'})
        return fig_dev