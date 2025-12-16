# import pandas as pd
# from dash import dcc, html, Input, Output
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
#
#
# class ServicePerformanceDashboard:
#     """
#     A class-based dashboard for visualizing service performance metrics.
#     Can be integrated as a tab in a larger Dash application.
#     """
#
#     def __init__(self, csv_path="services_weekly.csv"):
#         """
#         Initialize the dashboard with data from CSV.
#
#         Args:
#             csv_path: Path to the services_weekly.csv file
#         """
#         self.csv_path = csv_path
#         self.df = None
#         self.services = None
#         self.min_week = None
#         self.max_week = None
#
#         # Shape palette
#         self.shapes = {
#             "emergency": "circle",
#             "surgery": "square",
#             "ICU": "diamond",
#             "general_medicine": "cross",
#             "pediatrics": "x",
#             "cardiology": "triangle-up",
#             "orthopedics": "triangle-down",
#             "radiology": "pentagon",
#             "laboratory": "hexagon",
#             "pharmacy": "star"
#         }
#
#         # Quadrant colors
#         self.quadrant_colors = {
#             "low_refusal_high_sat": "rgb(46, 134, 193)",
#             "high_refusal_high_sat": "rgb(255, 152, 0)",
#             "low_refusal_low_sat": "rgb(76, 175, 80)",
#             "high_refusal_low_sat": "rgb(156, 39, 176)"
#         }
#
#         # Styles
#         self.page_style = {
#             "fontFamily": "Inter, Arial, sans-serif",
#             "padding": "30px",
#             "backgroundColor": "#f7f9fb"
#         }
#
#         self.card_style = {
#             "background": "white",
#             "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
#             "borderRadius": "10px",
#             "padding": "15px",
#             "width": "48%",
#             "marginBottom": "15px",
#             "borderLeft": "5px solid",
#         }
#
#         self.header_style = {
#             "textAlign": "center",
#             "marginBottom": "25px",
#             "color": "#2c3e50"
#         }
#
#         self.slider_container_style = {
#             "background": "white",
#             "borderRadius": "12px",
#             "padding": "20px",
#             "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
#             "marginBottom": "25px",
#             "width": "480px",
#             "marginLeft": "auto",
#             "marginRight": "auto",
#         }
#
#         # Load data
#         self._load_data()
#
#     def _load_data(self):
#         """Load and preprocess data from CSV."""
#         try:
#             self.df = pd.read_csv(self.csv_path)
#         except FileNotFoundError as e:
#             print(f"Error loading CSV file: {e}. Please ensure {self.csv_path} is in the directory.")
#             raise
#
#         # Normalize satisfaction 0–100 → 1–5
#         self.df["satisfaction_5pt"] = 1 + 4 * (self.df["patient_satisfaction"] / 100)
#
#         self.services = self.df["service"].unique()
#         self.min_week = int(self.df["week"].min())
#         self.max_week = int(self.df["week"].max())
#
#     def get_layout(self):
#         """
#         Return the layout for this dashboard.
#         This can be used as content for a dcc.Tab or a page in a multi-page app.
#         """
#         return html.Div([
#             html.H2("Service Performance Dashboard", style=self.header_style),
#
#             # Week range selector
#             html.Div([
#                 html.Label("Select Week Range", style={"fontWeight": "bold"}),
#                 dcc.RangeSlider(
#                     id="week-range",
#                     min=self.min_week,
#                     max=self.max_week,
#                     step=1,
#                     value=[self.min_week, self.max_week],
#                     allowCross=False,
#                     marks={i: f"Week {i}" for i in range(
#                         self.min_week,
#                         self.max_week + 1,
#                         max(1, (self.max_week - self.min_week) // 4)
#                     )},
#                     tooltip={"always_visible": False, "placement": "bottom"}
#                 ),
#             ], style=self.slider_container_style),
#
#             # Bubble chart and histograms side by side
#             html.Div([
#                 html.Div([
#                     dcc.Graph(id="bubble-chart"),
#                 ], style={"width": "65%", "display": "inline-block", "verticalAlign": "top"}),
#
#                 html.Div([
#                     html.Label("Select Service for Histograms",
#                                style={"fontWeight": "bold", "marginBottom": "10px"}),
#                     dcc.Dropdown(
#                         id="service-dropdown",
#                         options=[{"label": s, "value": s} for s in self.services],
#                         value=self.services[0],
#                         clearable=False,
#                         style={"marginBottom": "15px"}
#                     ),
#
#                     # Slider for Refusal Rate histogram bins
#                     html.Label("Refusal Rate Bucket Size (%)",
#                                style={"fontWeight": "bold", "marginTop": "10px"}),
#                     dcc.Slider(
#                         id="refusal-bins-slider",
#                         min=1,
#                         max=20,
#                         step=1,
#                         value=5,
#                         marks={i: f"{i}%" for i in [1, 5, 10, 15, 20]},
#                         tooltip={"always_visible": False, "placement": "bottom"}
#                     ),
#
#                     # Slider for Satisfaction histogram bins
#                     html.Label("Satisfaction Bucket Size",
#                                style={"fontWeight": "bold", "marginTop": "15px"}),
#                     dcc.Slider(
#                         id="satisfaction-bins-slider",
#                         min=0.1,
#                         max=1.0,
#                         step=0.1,
#                         value=0.2,
#                         marks={i / 10: f"{i / 10:.1f}" for i in [1, 2, 5, 10]},
#                         tooltip={"always_visible": False, "placement": "bottom"}
#                     ),
#
#                     dcc.Graph(id="histogram-charts"),
#                 ], style={"width": "33%", "display": "inline-block",
#                           "verticalAlign": "top", "marginLeft": "2%"}),
#             ], style={"marginBottom": "30px"}),
#
#             # Quadrant descriptions
#             self._get_quadrant_descriptions()
#
#         ], style=self.page_style)
#
#     def _get_quadrant_descriptions(self):
#         """Create the quadrant description cards."""
#         return html.Div([
#             # Top row
#             html.Div([
#                 html.Div([
#                     html.H4("Low Refusal • High Satisfaction",
#                             style={"color": self.quadrant_colors["low_refusal_high_sat"]}),
#                     html.P("Best performing services providing excellent access and quality.")
#                 ], style={**self.card_style,
#                           "borderLeftColor": self.quadrant_colors["low_refusal_high_sat"],
#                           "width": "48%"}),
#
#                 html.Div([
#                     html.H4("High Refusal • High Satisfaction",
#                             style={"color": self.quadrant_colors["high_refusal_high_sat"]}),
#                     html.P("Strong perceived quality, but access or capacity constraints exist.")
#                 ], style={**self.card_style,
#                           "borderLeftColor": self.quadrant_colors["high_refusal_high_sat"],
#                           "width": "48%"}),
#             ], style={"display": "flex", "justifyContent": "space-between",
#                       "marginBottom": "15px"}),
#
#             # Bottom row
#             html.Div([
#                 html.Div([
#                     html.H4("Low Refusal • Low Satisfaction",
#                             style={"color": self.quadrant_colors["low_refusal_low_sat"]}),
#                     html.P("Patients are accessing services, but quality improvements are needed.")
#                 ], style={**self.card_style,
#                           "borderLeftColor": self.quadrant_colors["low_refusal_low_sat"],
#                           "width": "48%"}),
#
#                 html.Div([
#                     html.H4("High Refusal • Low Satisfaction",
#                             style={"color": self.quadrant_colors["high_refusal_low_sat"]}),
#                     html.P("Critical zone: both access and quality need urgent attention.")
#                 ], style={**self.card_style,
#                           "borderLeftColor": self.quadrant_colors["high_refusal_low_sat"],
#                           "width": "48%"}),
#             ], style={"display": "flex", "justifyContent": "space-between"}),
#
#         ], style={"marginTop": "35px"})
#
#     def register_callbacks(self, app):
#         """
#         Register callbacks with the Dash app.
#
#         Args:
#             app: The Dash app instance
#         """
#
#         @app.callback(
#             [Output("bubble-chart", "figure"),
#              Output("histogram-charts", "figure")],
#             [Input("week-range", "value"),
#              Input("service-dropdown", "value"),
#              Input("refusal-bins-slider", "value"),
#              Input("satisfaction-bins-slider", "value")]
#         )
#         def update_chart(week_range, selected_service, refusal_bucket_size, satisfaction_bucket_size):
#             return self._update_chart(week_range, selected_service,
#                                       refusal_bucket_size, satisfaction_bucket_size)
#
#     def _update_chart(self, week_range, selected_service, refusal_bucket_size, satisfaction_bucket_size):
#         """Internal method to update charts based on selections."""
#         start_week, end_week = week_range
#
#         # Filter main dataset
#         dff = self.df[(self.df["week"] >= start_week) & (self.df["week"] <= end_week)]
#
#         if dff.empty:
#             empty_layout = {
#                 "title": "No Data Available for Selected Week Range",
#                 "xaxis": {"visible": False},
#                 "yaxis": {"visible": False},
#                 "height": 700,
#             }
#             return go.Figure(layout=empty_layout), go.Figure(layout=empty_layout)
#
#         # Create bubble chart
#         bubble_fig = self._create_bubble_chart(dff, start_week, end_week)
#
#         # Create histogram charts
#         hist_fig = self._create_histogram_charts(dff, selected_service,
#                                                  refusal_bucket_size, satisfaction_bucket_size)
#
#         return bubble_fig, hist_fig
#
#     def _create_bubble_chart(self, dff, start_week, end_week):
#         """Create the bubble chart visualization."""
#         # Aggregate by service
#         agg = dff.groupby("service", as_index=False).agg({
#             "patients_request": "sum",
#             "patients_refused": "sum",
#             "satisfaction_5pt": "mean"
#         })
#         agg["refusal_rate"] = agg["patients_refused"] / agg["patients_request"]
#
#         fig = go.Figure()
#
#         # Add background shading
#         x_mid, y_mid = 0.50, 2.50
#
#         fig.add_shape(type="rect", x0=0, x1=x_mid, y0=y_mid, y1=5,
#                       fillcolor="rgba(230,242,250,0.6)", line_width=0, layer="below")
#         fig.add_shape(type="rect", x0=x_mid, x1=1, y0=y_mid, y1=5,
#                       fillcolor="rgba(255,240,224,0.6)", line_width=0, layer="below")
#         fig.add_shape(type="rect", x0=0, x1=x_mid, y0=0, y1=y_mid,
#                       fillcolor="rgba(233,245,234,0.6)", line_width=0, layer="below")
#         fig.add_shape(type="rect", x0=x_mid, x1=1, y0=0, y1=y_mid,
#                       fillcolor="rgba(241,233,246,0.6)", line_width=0, layer="below")
#
#         # Add quadrant guide lines
#         fig.add_hline(y=y_mid, line_dash="dot", line_color="#888", layer="below")
#         fig.add_vline(x=x_mid, line_dash="dot", line_color="#888", layer="below")
#
#         # Add scatter traces
#         for _, row in agg.iterrows():
#             fig.add_trace(go.Scatter(
#                 x=[row["refusal_rate"]],
#                 y=[row["satisfaction_5pt"]],
#                 mode="markers+text",
#                 text=[row["service"]],
#                 textposition="top center",
#                 marker=dict(
#                     size=18,
#                     color="black",
#                     symbol=self.shapes.get(row["service"], "circle"),
#                     opacity=0.92,
#                     line=dict(width=1, color="white")
#                 ),
#                 name=row["service"]
#             ))
#
#         # Update layout
#         fig.update_xaxes(
#             range=[-0.02, 1.02],
#             tickformat=".0%",
#             title="Refusal Rate (%)",
#             zeroline=False,
#             gridcolor="rgba(200,200,200,0.2)"
#         )
#         fig.update_yaxes(
#             range=[-0.15, 5.15],
#             title="Patient Satisfaction (1–5)",
#             gridcolor="rgba(200,200,200,0.2)"
#         )
#         fig.update_layout(
#             title={
#                 'text': f"Service Performance: Refusal Rate vs Satisfaction (Weeks {start_week}–{end_week})",
#                 'x': 0.5, 'xanchor': 'center'
#             },
#             height=700,
#             plot_bgcolor="#fcfcfc",
#             paper_bgcolor="#fcfcfc",
#             legend=dict(orientation="h", x=0.5, y=-0.15, xanchor="center"),
#             margin=dict(t=80, l=60, r=60, b=120)
#         )
#
#         return fig
#
#     def _create_histogram_charts(self, dff, selected_service, refusal_bucket_size, satisfaction_bucket_size):
#         """Create the histogram charts for a selected service."""
#         service_data = dff[dff["service"] == selected_service]
#
#         if service_data.empty:
#             empty_hist_layout = {
#                 "title": f"No Data for {selected_service}",
#                 "xaxis": {"visible": False},
#                 "yaxis": {"visible": False},
#                 "height": 700,
#             }
#             return go.Figure(layout=empty_hist_layout)
#
#         # Calculate refusal rate
#         service_data = service_data.copy()
#         service_data["refusal_rate"] = service_data["patients_refused"] / service_data["patients_request"]
#         service_data["refusal_rate_pct"] = service_data["refusal_rate"] * 100
#
#         fig_hist = make_subplots(
#             rows=2, cols=1,
#             subplot_titles=(f"{selected_service}: Refusal Rate",
#                             f"{selected_service}: Satisfaction"),
#             vertical_spacing=0.15
#         )
#
#         # Histogram 1: Refusal Rate
#         fig_hist.add_trace(
#             go.Histogram(
#                 x=service_data["refusal_rate_pct"],
#                 xbins=dict(start=0, end=100, size=refusal_bucket_size),
#                 marker_color="black",
#                 opacity=0.7,
#                 name="Refusal Rate"
#             ),
#             row=1, col=1
#         )
#
#         # Histogram 2: Satisfaction
#         fig_hist.add_trace(
#             go.Histogram(
#                 x=service_data["satisfaction_5pt"],
#                 xbins=dict(start=1, end=5, size=satisfaction_bucket_size),
#                 marker_color="black",
#                 opacity=0.7,
#                 name="Satisfaction"
#             ),
#             row=2, col=1
#         )
#
#         fig_hist.update_xaxes(title_text="Refusal Rate (%)", range=[0, 100], row=1, col=1)
#         fig_hist.update_xaxes(title_text="Satisfaction Score (1-5)", range=[1, 5], row=2, col=1)
#         fig_hist.update_yaxes(title_text="Count", row=1, col=1)
#         fig_hist.update_yaxes(title_text="Count", row=2, col=1)
#
#         fig_hist.update_layout(
#             height=700,
#             showlegend=False,
#             plot_bgcolor="#fcfcfc",
#             paper_bgcolor="#fcfcfc",
#             margin=dict(t=60, l=60, r=30, b=60)
#         )
#
#         return fig_hist

import pandas as pd
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ServicePerformanceDashboard:
    """
    A class-based dashboard for visualizing service performance metrics.
    Can be integrated as a tab in a larger Dash application.
    """

    def __init__(self, csv_path="services_weekly.csv", id_prefix=""):
        """
        Initialize the dashboard with data from CSV.

        Args:
            csv_path: Path to the services_weekly.csv file
            id_prefix: Prefix for all component IDs to avoid conflicts
        """
        self.csv_path = csv_path
        self.id_prefix = id_prefix
        self.df = None
        self.services = None
        self.min_week = None
        self.max_week = None

        # Shape palette
        self.shapes = {
            "emergency": "circle",
            "surgery": "square",
            "ICU": "diamond",
            "general_medicine": "cross",
            "pediatrics": "x",
            "cardiology": "triangle-up",
            "orthopedics": "triangle-down",
            "radiology": "pentagon",
            "laboratory": "hexagon",
            "pharmacy": "star"
        }

        # Quadrant colors
        self.quadrant_colors = {
            "low_refusal_high_sat": "rgb(46, 134, 193)",
            "high_refusal_high_sat": "rgb(255, 152, 0)",
            "low_refusal_low_sat": "rgb(76, 175, 80)",
            "high_refusal_low_sat": "rgb(156, 39, 176)"
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
        try:
            self.df = pd.read_csv(self.csv_path)
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}. Please ensure {self.csv_path} is in the directory.")
            raise

        # Normalize satisfaction 0–100 → 1–5
        self.df["satisfaction_5pt"] = 1 + 4 * (self.df["patient_satisfaction"] / 100)

        self.services = self.df["service"].unique()
        self.min_week = int(self.df["week"].min())
        self.max_week = int(self.df["week"].max())

    def get_layout(self):
        """
        Return the layout for this dashboard.
        This can be used as content for a dcc.Tab or a page in a multi-page app.
        """
        return html.Div([
            html.H2("Service Performance Dashboard", style=self.header_style),

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

            # Bubble chart and histograms side by side
            html.Div([
                html.Div([
                    dcc.Graph(id=self._get_id("bubble-chart")),
                ], style={"width": "65%", "display": "inline-block", "verticalAlign": "top"}),

                html.Div([
                    html.Label("Select Service for Histograms",
                               style={"fontWeight": "bold", "marginBottom": "10px"}),
                    dcc.Dropdown(
                        id=self._get_id("service-dropdown"),
                        options=[{"label": s, "value": s} for s in self.services],
                        value=self.services[0],
                        clearable=False,
                        style={"marginBottom": "15px"}
                    ),

                    # Slider for Refusal Rate histogram bins
                    html.Label("Refusal Rate Bucket Size (%)",
                               style={"fontWeight": "bold", "marginTop": "10px"}),
                    dcc.Slider(
                        id=self._get_id("refusal-bins-slider"),
                        min=1,
                        max=20,
                        step=1,
                        value=5,
                        marks={i: f"{i}%" for i in [1, 5, 10, 15, 20]},
                        tooltip={"always_visible": False, "placement": "bottom"}
                    ),

                    # Slider for Satisfaction histogram bins
                    html.Label("Satisfaction Bucket Size",
                               style={"fontWeight": "bold", "marginTop": "15px"}),
                    dcc.Slider(
                        id=self._get_id("satisfaction-bins-slider"),
                        min=0.1,
                        max=1.0,
                        step=0.1,
                        value=0.2,
                        marks={i / 10: f"{i / 10:.1f}" for i in [1, 2, 5, 10]},
                        tooltip={"always_visible": False, "placement": "bottom"}
                    ),

                    dcc.Graph(id=self._get_id("histogram-charts")),
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
            [Output(self._get_id("bubble-chart"), "figure"),
             Output(self._get_id("histogram-charts"), "figure")],
            [Input(self._get_id("week-range"), "value"),
             Input(self._get_id("service-dropdown"), "value"),
             Input(self._get_id("refusal-bins-slider"), "value"),
             Input(self._get_id("satisfaction-bins-slider"), "value")]
        )
        def update_chart(week_range, selected_service, refusal_bucket_size, satisfaction_bucket_size):
            return self._update_chart(week_range, selected_service,
                                      refusal_bucket_size, satisfaction_bucket_size)

    def _update_chart(self, week_range, selected_service, refusal_bucket_size, satisfaction_bucket_size):
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

        # Create bubble chart
        bubble_fig = self._create_bubble_chart(dff, start_week, end_week)

        # Create histogram charts
        hist_fig = self._create_histogram_charts(dff, selected_service,
                                                 refusal_bucket_size, satisfaction_bucket_size)

        return bubble_fig, hist_fig

    def _create_bubble_chart(self, dff, start_week, end_week):
        """Create the bubble chart visualization."""
        # Aggregate by service
        agg = dff.groupby("service", as_index=False).agg({
            "patients_request": "sum",
            "patients_refused": "sum",
            "satisfaction_5pt": "mean"
        })
        agg["refusal_rate"] = agg["patients_refused"] / agg["patients_request"]

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

        # Add scatter traces
        for _, row in agg.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["refusal_rate"]],
                y=[row["satisfaction_5pt"]],
                mode="markers+text",
                text=[row["service"]],
                textposition="top center",
                marker=dict(
                    size=18,
                    color="black",
                    symbol=self.shapes.get(row["service"], "circle"),
                    opacity=0.92,
                    line=dict(width=1, color="white")
                ),
                name=row["service"]
            ))

        # Update layout
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
                'text': f"Service Performance: Refusal Rate vs Satisfaction (Weeks {start_week}–{end_week})",
                'x': 0.5, 'xanchor': 'center'
            },
            height=700,
            plot_bgcolor="#fcfcfc",
            paper_bgcolor="#fcfcfc",
            legend=dict(orientation="h", x=0.5, y=-0.15, xanchor="center"),
            margin=dict(t=80, l=60, r=60, b=120)
        )

        return fig

    def _create_histogram_charts(self, dff, selected_service, refusal_bucket_size, satisfaction_bucket_size):
        """Create the histogram charts for a selected service."""
        service_data = dff[dff["service"] == selected_service]

        if service_data.empty:
            empty_hist_layout = {
                "title": f"No Data for {selected_service}",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "height": 700,
            }
            return go.Figure(layout=empty_hist_layout)

        # Calculate refusal rate
        service_data = service_data.copy()
        service_data["refusal_rate"] = service_data["patients_refused"] / service_data["patients_request"]
        service_data["refusal_rate_pct"] = service_data["refusal_rate"] * 100

        fig_hist = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f"{selected_service}: Refusal Rate",
                            f"{selected_service}: Satisfaction"),
            vertical_spacing=0.15
        )

        # Histogram 1: Refusal Rate
        fig_hist.add_trace(
            go.Histogram(
                x=service_data["refusal_rate_pct"],
                xbins=dict(start=0, end=100, size=refusal_bucket_size),
                marker_color="black",
                opacity=0.7,
                name="Refusal Rate"
            ),
            row=1, col=1
        )

        # Histogram 2: Satisfaction
        fig_hist.add_trace(
            go.Histogram(
                x=service_data["satisfaction_5pt"],
                xbins=dict(start=1, end=5, size=satisfaction_bucket_size),
                marker_color="black",
                opacity=0.7,
                name="Satisfaction"
            ),
            row=2, col=1
        )

        fig_hist.update_xaxes(title_text="Refusal Rate (%)", range=[0, 100], row=1, col=1)
        fig_hist.update_xaxes(title_text="Satisfaction Score (1-5)", range=[1, 5], row=2, col=1)
        fig_hist.update_yaxes(title_text="Count", row=1, col=1)
        fig_hist.update_yaxes(title_text="Count", row=2, col=1)

        fig_hist.update_layout(
            height=700,
            showlegend=False,
            plot_bgcolor="#fcfcfc",
            paper_bgcolor="#fcfcfc",
            margin=dict(t=60, l=60, r=30, b=60)
        )

        return fig_hist