# Import necessary libraries
import pandas as pd
import numpy as np
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde


class ServicePerformanceDashboard:
    """
    Class for visualizing service performance metrics.
    (meant to be used as a single tab in the app script)
    """

    def __init__(self, csv_path="services_weekly.csv", id_prefix=""):
        """
        Initializes the dashboard with data from CSV.

        Args:
            csv_path: Path to the needed .csv file
            id_prefix: Prefix for all component IDs to avoid conflicts when having multiple tabs
        """
        self.csv_path = csv_path
        self.id_prefix = id_prefix
        self.df = None
        self.services = None
        self.min_week = None
        self.max_week = None

        # Shape representing each type of service in the scatter plot
        self.shapes = {
            "emergency": "circle",
            "surgery": "square",
            "ICU": "diamond",
            "general_medicine": "cross",
        }

        # Quadrant background colors
        self.quadrant_colors = {
            "low_refusal_high_sat": "rgb(46, 134, 193)",
            "high_refusal_high_sat": "rgb(255, 152, 0)",
            "low_refusal_low_sat": "rgb(76, 175, 80)",
            "high_refusal_low_sat": "rgb(156, 39, 176)"
        }

        # Assign each service a quadrant color for KDE
        self.service_kde_colors = {
            "emergency": "rgb(46, 134, 193)",
            "surgery": "rgb(255, 152, 0)",
            "ICU": "rgb(76, 175, 80)",
            "general_medicine": "rgb(156, 39, 176)",
        }

        # Styles
        self.page_style = {
            "fontFamily": "Inter, Arial, sans-serif",
            "padding": "30px",
            "backgroundColor": "#f7f9fb"
        }

        self.card_style = {
            "background": "white",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
            "borderRadius": "10px",
            "padding": "15px",
            "width": "48%",
            "marginBottom": "15px",
            "borderLeft": "5px solid",
        }

        self.header_style = {
            "textAlign": "center",
            "marginBottom": "25px",
            "color": "#2c3e50"
        }

        self.slider_container_style = {
            "background": "white",
            "borderRadius": "12px",
            "padding": "20px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
            "marginBottom": "25px",
            "width": "480px",
            "marginLeft": "auto",
            "marginRight": "auto",
        }

        # Load data
        self._load_data()

    def _get_id(self, component_id):
        """Helper method to create unique IDs with prefix."""
        return f"{self.id_prefix}{component_id}" if self.id_prefix else component_id

    def _load_data(self):
        """Load and preprocess data from CSV."""
        # Try block to handle exceptions properly
        try:
            self.df = pd.read_csv(self.csv_path)
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}. File {self.csv_path} not found.")
            raise

        # Normalize satisfaction from 0–100 to 1–5 range
        self.df["satisfaction_5pt"] = 1 + 4 * (self.df["patient_satisfaction"] / 100)

        self.services = self.df["service"].unique()
        self.min_week = int(self.df["week"].min())
        self.max_week = int(self.df["week"].max())

    def _get_quadrant_color(self, refusal_rate, satisfaction):
        """
        Determine which quadrant a service belongs to and return the corresponding color.

        Args:
            refusal_rate: Service refusal rate (0-1)
            satisfaction: Service satisfaction score (1-5)

        Returns:
            RGB color string for the quadrant
        """
        x_mid, y_mid = 0.50, 2.50

        # low refusal, high satisfaction
        if refusal_rate < x_mid and satisfaction >= y_mid:
            return self.quadrant_colors["low_refusal_high_sat"]
        # high refusal, high satisfaction
        elif refusal_rate >= x_mid and satisfaction >= y_mid:
            return self.quadrant_colors["high_refusal_high_sat"]
        # low refusal, low satisfaction
        elif refusal_rate < x_mid and satisfaction < y_mid:
            return self.quadrant_colors["low_refusal_low_sat"]
        else:  # high refusal, low satisfaction
            return self.quadrant_colors["high_refusal_low_sat"]

    def get_layout(self):
        """
        Return the layout for this dashboard.
        """
        return html.Div([
            
            # Hidden store for tracking visible services
            dcc.Store(id=self._get_id("visible-services-store"), data=None),

            # Week range selector
            html.Div([
                html.Label("Select Week Range", style={"fontWeight": "bold"}),
                dcc.RangeSlider(
                    id=self._get_id("week-range"),
                    min=self.min_week,
                    max=self.max_week,
                    step=1,
                    value=[self.min_week, self.max_week],
                    allowCross=False,
                    marks={i: f"Week {i}" for i in range(
                        self.min_week,
                        self.max_week + 1,
                        max(1, (self.max_week - self.min_week) // 4)
                    )},
                    tooltip={"always_visible": False, "placement": "bottom"}
                ),
            ], style=self.slider_container_style),

            # Bubble chart and KDE plots side by side
            html.Div([
                html.Div([
                    dcc.Graph(id=self._get_id("bubble-chart")),
                ], style={"width": "65%", "display": "inline-block", "verticalAlign": "top"}),

                html.Div([
                    # Slider for Refusal Rate KDE bandwidth
                    html.Label("Refusal Rate KDE Bandwidth",
                               style={"fontWeight": "bold", "marginTop": "10px"}),
                    dcc.Slider(
                        id=self._get_id("kde-bandwidth-refusal"),
                        min=0.01,
                        max=0.5,
                        step=0.01,
                        value=0.1,
                        marks={i / 100: f"{i / 100:.2f}" for i in [1, 10, 20, 30, 40, 50]},
                        tooltip={"always_visible": False, "placement": "bottom"}
                    ),

                    # Slider for Satisfaction KDE bandwidth
                    html.Label("Satisfaction KDE Bandwidth",
                               style={"fontWeight": "bold", "marginTop": "20px"}),
                    dcc.Slider(
                        id=self._get_id("kde-bandwidth-satisfaction"),
                        min=0.01,
                        max=0.5,
                        step=0.01,
                        value=0.1,
                        marks={i / 100: f"{i / 100:.2f}" for i in [1, 10, 20, 30, 40, 50]},
                        tooltip={"always_visible": False, "placement": "bottom"}
                    ),

                    dcc.Graph(id=self._get_id("kde-charts"), config={'displayModeBar': False}, animate=True),
                ], style={"width": "33%", "display": "inline-block",
                          "verticalAlign": "top", "marginLeft": "2%"}),
            ], style={"marginBottom": "30px"}),

            # Quadrant descriptions
            self._get_quadrant_descriptions()

        ], style=self.page_style)

    def _get_quadrant_descriptions(self):
        """Create the quadrant description cards."""
        return html.Div([
            # Top row
            html.Div([
                html.Div([
                    html.H4("Low Refusal • High Satisfaction",
                            style={"color": self.quadrant_colors["low_refusal_high_sat"]}),
                    html.P("Best performing services providing excellent access and quality.")
                ], style={**self.card_style,
                          "borderLeftColor": self.quadrant_colors["low_refusal_high_sat"],
                          "width": "48%"}),

                html.Div([
                    html.H4("High Refusal • High Satisfaction",
                            style={"color": self.quadrant_colors["high_refusal_high_sat"]}),
                    html.P("Strong perceived quality, but access or capacity constraints exist.")
                ], style={**self.card_style,
                          "borderLeftColor": self.quadrant_colors["high_refusal_high_sat"],
                          "width": "48%"}),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "marginBottom": "15px"}),

            # Bottom row
            html.Div([
                html.Div([
                    html.H4("Low Refusal • Low Satisfaction",
                            style={"color": self.quadrant_colors["low_refusal_low_sat"]}),
                    html.P("Patients are accessing services, but quality improvements are needed.")
                ], style={**self.card_style,
                          "borderLeftColor": self.quadrant_colors["low_refusal_low_sat"],
                          "width": "48%"}),

                html.Div([
                    html.H4("High Refusal • Low Satisfaction",
                            style={"color": self.quadrant_colors["high_refusal_low_sat"]}),
                    html.P("Critical zone: both access and quality need urgent attention.")
                ], style={**self.card_style,
                          "borderLeftColor": self.quadrant_colors["high_refusal_low_sat"],
                          "width": "48%"}),
            ], style={"display": "flex", "justifyContent": "space-between"}),

        ], style={"marginTop": "35px"})

    def register_callbacks(self, app):
        """
        Register callbacks with the Dash app.

        Args:
            app: The Dash app instance
        """

        @app.callback(
            [Output(self._get_id("visible-services-store"), "data"),
             Output(self._get_id("kde-charts"), "figure")],
            [Input(self._get_id("bubble-chart"), "restyleData"),
             Input(self._get_id("week-range"), "value"),
             Input(self._get_id("kde-bandwidth-refusal"), "value"),
             Input(self._get_id("kde-bandwidth-satisfaction"), "value")],
            [State(self._get_id("visible-services-store"), "data")],
            prevent_initial_call=False
        )
        def update_kde_from_legend(restyle_data, week_range, kde_bw_refusal, kde_bw_satisfaction, stored_visibility):
            """Update KDE charts when legend items are clicked in bubble chart."""
            start_week, end_week = week_range
            dff = self.df[(self.df["week"] >= start_week) & (self.df["week"] <= end_week)]

            # Get list of all services
            agg = dff.groupby("service", as_index=False).agg({
                "patients_request": "sum",
                "patients_refused": "sum",
                "satisfaction_5pt": "mean"
            })
            agg["refusal_rate"] = agg["patients_refused"] / agg["patients_request"]
            all_services = agg["service"].tolist()

            # Initialize visibility state
            if stored_visibility is None:
                visible_services = {svc: True for svc in all_services}
            else:
                visible_services = stored_visibility.copy()
                # Add any new services that might have appeared
                for svc in all_services:
                    if svc not in visible_services:
                        visible_services[svc] = True

            # Update visibility based on legend clicks
            if restyle_data and len(restyle_data) == 2 and 'visible' in restyle_data[0]:
                visibility_values = restyle_data[0]['visible']
                trace_indices = restyle_data[1]

                # Handle single or multiple indices
                if not isinstance(trace_indices, list):
                    trace_indices = [trace_indices]
                if not isinstance(visibility_values, list):
                    visibility_values = [visibility_values]

                # Update visible services based on legend clicks
                for idx, vis in zip(trace_indices, visibility_values):
                    if idx < len(all_services):
                        service = all_services[idx]
                        # vis can be True, False, or 'legendonly'
                        visible_services[service] = (vis == True)

            # Get set of currently visible services
            visible_set = {svc for svc, is_visible in visible_services.items() if is_visible}

            kde_fig = self._create_kde_charts(dff, agg, kde_bw_refusal, kde_bw_satisfaction, visible_set)

            return visible_services, kde_fig

        @app.callback(
            Output(self._get_id("bubble-chart"), "figure"),
            [Input(self._get_id("week-range"), "value")]
        )
        def update_bubble(week_range):
            """Update bubble chart when week range changes."""
            start_week, end_week = week_range
            dff = self.df[(self.df["week"] >= start_week) & (self.df["week"] <= end_week)]

            if dff.empty:
                empty_layout = {
                    "title": "No Data Available for Selected Week Range",
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False},
                    "height": 700,
                }
                return go.Figure(layout=empty_layout)

            agg = dff.groupby("service", as_index=False).agg({
                "patients_request": "sum",
                "patients_refused": "sum",
                "satisfaction_5pt": "mean"
            })
            agg["refusal_rate"] = agg["patients_refused"] / agg["patients_request"]

            return self._create_bubble_chart(agg, start_week, end_week)

    def _update_chart(self, week_range, kde_bw_refusal, kde_bw_satisfaction, visible_services_data):
        """Internal method to update charts based on selections."""
        start_week, end_week = week_range

        # Filter main dataset
        dff = self.df[(self.df["week"] >= start_week) & (self.df["week"] <= end_week)]

        if dff.empty:
            empty_layout = {
                "title": "No Data Available for Selected Week Range",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "height": 700,
            }
            return go.Figure(layout=empty_layout), go.Figure(layout=empty_layout)

        # Aggregate by service
        agg = dff.groupby("service", as_index=False).agg({
            "patients_request": "sum",
            "patients_refused": "sum",
            "satisfaction_5pt": "mean"
        })
        agg["refusal_rate"] = agg["patients_refused"] / agg["patients_request"]

        # Determine visible services (extract from figure state if available)
        visible_services = set(agg["service"].tolist())

        # Create bubble chart
        bubble_fig = self._create_bubble_chart(agg, start_week, end_week)

        # Create KDE charts (only show services visible in bubble chart)
        kde_fig = self._create_kde_charts(dff, agg, kde_bw_refusal, kde_bw_satisfaction, visible_services)

        return bubble_fig, kde_fig

    def _create_bubble_chart(self, agg, start_week, end_week):
        """Create the bubble chart visualization with smooth animations."""
        fig = go.Figure()

        # Add background shading
        x_mid, y_mid = 0.50, 2.50

        fig.add_shape(type="rect", x0=0, x1=x_mid, y0=y_mid, y1=5,
                      fillcolor="rgba(230,242,250,0.6)", line_width=0, layer="below")
        fig.add_shape(type="rect", x0=x_mid, x1=1, y0=y_mid, y1=5,
                      fillcolor="rgba(255,240,224,0.6)", line_width=0, layer="below")
        fig.add_shape(type="rect", x0=0, x1=x_mid, y0=0, y1=y_mid,
                      fillcolor="rgba(233,245,234,0.6)", line_width=0, layer="below")
        fig.add_shape(type="rect", x0=x_mid, x1=1, y0=0, y1=y_mid,
                      fillcolor="rgba(241,233,246,0.6)", line_width=0, layer="below")

        # Add quadrant guide lines
        fig.add_hline(y=y_mid, line_dash="dot", line_color="#888", layer="below")
        fig.add_vline(x=x_mid, line_dash="dot", line_color="#888", layer="below")

        # Add scatter traces with BLACK markers
        for _, row in agg.iterrows():
            service = row["service"]

            fig.add_trace(go.Scatter(
                x=[row["refusal_rate"]],
                y=[row["satisfaction_5pt"]],
                mode="markers+text",
                text=[service],
                textposition="top center",
                marker=dict(
                    size=18,
                    color="black",
                    symbol=self.shapes.get(service, "circle"),
                    opacity=0.92,
                    line=dict(width=1, color="white")
                ),
                customdata=[service],
                name=service,
                visible=True
            ))

        # Update layout with animation settings
        fig.update_xaxes(
            range=[-0.02, 1.02],
            tickformat=".0%",
            title="Refusal Rate (%)",
            zeroline=False,
            gridcolor="rgba(200,200,200,0.2)"
        )
        fig.update_yaxes(
            range=[-0.15, 5.15],
            title="Patient Satisfaction (1–5)",
            gridcolor="rgba(200,200,200,0.2)"
        )

        fig.update_layout(
            title={
                'text': f"Service Performance: Refusal Rate vs Satisfaction (Weeks {start_week}–{end_week})<br><sub>Click legend items to show/hide services</sub>",
                'x': 0.5, 'xanchor': 'center'
            },
            height=700,
            plot_bgcolor="#fcfcfc",
            paper_bgcolor="#fcfcfc",
            legend=dict(orientation="h", x=0.5, y=-0.15, xanchor="center"),
            margin=dict(t=80, l=60, r=60, b=120),
            transition={
                'duration': 500,
                'easing': 'cubic-in-out'
            }
        )

        return fig

    def _create_kde_charts(self, dff, agg, kde_bw_refusal, kde_bw_satisfaction, visible_services):
        """Create overlapping KDE plots for visible services only with assigned quadrant colors."""
        # Calculate refusal rate for all data
        dff = dff.copy()
        dff["refusal_rate"] = dff["patients_refused"] / dff["patients_request"]
        dff["refusal_rate_pct"] = dff["refusal_rate"] * 100

        fig_kde = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Refusal Rate Distribution (Visible Services)",
                            "Satisfaction Distribution (Visible Services)"),
            vertical_spacing=0.15
        )

        # Only show services that are in visible_services
        services_to_show = [s for s in visible_services if s in dff["service"].unique()]

        # Track max y-values for proper scaling
        max_y_refusal = 0
        max_y_satisfaction = 0

        # KDE Plot 1: Refusal Rate
        for service in services_to_show:
            service_data = dff[dff["service"] == service]["refusal_rate_pct"].dropna()

            if len(service_data) > 1:
                try:
                    kde = gaussian_kde(service_data, bw_method=kde_bw_refusal)
                    x_range = np.linspace(0, 100, 200)
                    kde_values = kde(x_range)

                    # Track maximum y-value
                    max_y_refusal = max(max_y_refusal, np.max(kde_values))

                    fig_kde.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=kde_values,
                            mode="lines",
                            name=service,
                            line=dict(
                                color=self.service_kde_colors.get(service, "rgb(128, 128, 128)"),
                                width=2
                            ),
                            legendgroup=service,
                            showlegend=True
                        ),
                        row=1, col=1
                    )
                except:
                    pass

        # KDE Plot 2: Satisfaction
        for service in services_to_show:
            service_data = dff[dff["service"] == service]["satisfaction_5pt"].dropna()

            if len(service_data) > 1:
                try:
                    kde = gaussian_kde(service_data, bw_method=kde_bw_satisfaction)
                    x_range = np.linspace(1, 5, 200)
                    kde_values = kde(x_range)

                    # Track maximum y-value
                    max_y_satisfaction = max(max_y_satisfaction, np.max(kde_values))

                    fig_kde.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=kde_values,
                            mode="lines",
                            name=service,
                            line=dict(
                                color=self.service_kde_colors.get(service, "rgb(128, 128, 128)"),
                                width=2
                            ),
                            legendgroup=service,
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                except:
                    pass

        # Update axes with dynamic y-ranges (add 10% padding)
        fig_kde.update_xaxes(title_text="Refusal Rate (%)", range=[0, 100], row=1, col=1)
        fig_kde.update_xaxes(title_text="Satisfaction Score (1-5)", range=[1, 5], row=2, col=1)

        fig_kde.update_yaxes(
            title_text="Density",
            range=[0, max_y_refusal * 1.1] if max_y_refusal > 0 else [0, 1],
            row=1, col=1
        )
        fig_kde.update_yaxes(
            title_text="Density",
            range=[0, max_y_satisfaction * 1.1] if max_y_satisfaction > 0 else [0, 1],
            row=2, col=1
        )

        fig_kde.update_layout(
            height=700,
            plot_bgcolor="#fcfcfc",
            paper_bgcolor="#fcfcfc",
            margin=dict(t=60, l=60, r=30, b=60),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02
            ),
            uirevision='constant'
        )

        # Add frame-based animation
        fig_kde.layout.updatemenus = [
            dict(
                type="buttons",
                showactive=False,
                visible=False,
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, {"frame": {"duration": 500, "redraw": True},
                                           "fromcurrent": True,
                                           "transition": {"duration": 500, "easing": "cubic-in-out"}}])]
            )
        ]

        return fig_kde