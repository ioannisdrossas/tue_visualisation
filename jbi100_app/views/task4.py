import pandas as pd
import numpy as np
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.colors import sample_colorscale


class CorrelationsDashboard:

    def __init__(self, services_csv="services_weekly.csv", schedule_csv="staff_schedule.csv", id_prefix=""):
        """
        Initialize the dashboard with data from CSV files.

        Args:
            services_csv: Path to the services_weekly.csv file
            schedule_csv: Path to the staff_schedule.csv file
            id_prefix: Prefix for all component IDs to avoid conflicts
        """
        self.services_csv = services_csv
        self.schedule_csv = schedule_csv
        self.id_prefix = id_prefix
        self.df = None
        self.services = None
        self.week_min = None
        self.week_max = None

        #histogram bin edges
        self.SAT_BINS = np.arange(0, 110, 10)
        self.SAT_BIN_CENTERS = (self.SAT_BINS[:-1] + self.SAT_BINS[1:]) / 2.0

        #PCP dimensions (exactly 4 attributes)
        self.PCP_DIMS = [
            ("Staff Morale", "staff_morale"),
            ("Available Beds", "available_beds"),
            ("Patient-to-Nurse Ratio", "patient_to_nurse_ratio"),
            ("Patient-to-Doctor Ratio", "patient_to_doctor_ratio"),
        ]

        #animation settings (ms)
        self.PCP_TRANSITION_MS = 350

        #styles
        self.page_style = {
            "fontFamily": "Inter, Arial, sans-serif",
            "padding": "20px",
            "backgroundColor": "white",
            "minHeight": "100vh",
        }

        self.header_style = {"marginBottom": "10px"}

        #load and prepare data
        self._load_data()

    def _get_id(self, component_id):
        """Helper method to create unique IDs with prefix."""
        return f"{self.id_prefix}{component_id}" if self.id_prefix else component_id

    def _load_data(self):
        """Load and preprocess data from CSV files."""
        try:
            df = pd.read_csv(self.services_csv)
            schedule = pd.read_csv(self.schedule_csv)
        except FileNotFoundError as e:
            print(f"Error loading CSV files: {e}")
            raise

#####################################################################

        #count nurses present per week and service
        nurse_counts = (
            schedule[schedule["role"].str.lower() == "nurse"]
            .groupby(["week", "service"])["present"]
            .sum()
            .reset_index(name="nurses_present")
        )

        #count doctors present per week and service
        doctor_counts = (
            schedule[schedule["role"].str.lower() == "doctor"]
            .groupby(["week", "service"])["present"]
            .sum()
            .reset_index(name="doctors_present")
        )

        # Add staff counts to the main service dataset
        df = df.merge(nurse_counts, on=["week", "service"], how="left")
        df = df.merge(doctor_counts, on=["week", "service"], how="left")

        # Replace zero staff counts to avoid division by zero
        df.loc[df["nurses_present"] == 0, "nurses_present"] = pd.NA
        df.loc[df["doctors_present"] == 0, "doctors_present"] = pd.NA

        # Compute patient-to-staff ratios
        df["patient_to_nurse_ratio"] = df["patients_admitted"] / df["nurses_present"]
        df["patient_to_doctor_ratio"] = df["patients_admitted"] / df["doctors_present"]

#####################################################################


        self.df = df

        #select the first four services to display
        self.services = sorted(df["service"].unique())[:4]

        # Define minimum and maximum weeks for the time range slider
        self.week_min = int(df["week"].min())
        self.week_max = int(df["week"].max())

    def get_layout(self):
        """
        Return the layout for this dashboard.
        This can be used as content for a dcc.Tab or a page in a multi-page app.
        """
        return html.Div([
            #store for selected satisfaction bin (for PCP only)
            dcc.Store(id=self._get_id("selected-sat-bin"), data=None),

            html.H2("Service Metrics Dashboard", style=self.header_style),

            #week range slider
            html.Div([
                html.Label("Time Range (weeks)"),
                dcc.RangeSlider(
                    id=self._get_id("week-range"),
                    min=self.week_min,
                    max=self.week_max,
                    step=1,
                    value=[self.week_min, self.week_max],
                    allowCross=False,
                    marks={w: str(w) for w in range(self.week_min, self.week_max + 1, 5)},
                    updatemode="mouseup",
                ),
            ], style={"marginBottom": "16px"}),

            #histogram under week bar
            html.Div([
                html.H4("Patient Satisfaction Histogram (Click a bin to filter PCP lines)"),
                dcc.Graph(id=self._get_id("satisfaction-hist")),
                html.Div(
                    "Tip: click a bar to select a bin; click it again to clear.",
                    style={"fontSize": "12px", "color": "#777"},
                ),
            ], style={"marginBottom": "20px"}),

            #PCP plots full width
            html.Div([
                html.H4("Parallel Coordinates (Color: Patient Satisfaction)"),
                html.Div([
                    html.Div(
                        dcc.Graph(
                            id=self._get_id(f"pcp-{self.services[0]}"),
                            animate=True,
                            animation_options={
                                "frame": {"duration": self.PCP_TRANSITION_MS, "redraw": True},
                                "transition": {"duration": self.PCP_TRANSITION_MS},
                            },
                        ),
                        style={"width": "50%", "display": "inline-block"},
                    ),
                    html.Div(
                        dcc.Graph(
                            id=self._get_id(f"pcp-{self.services[1]}"),
                            animate=True,
                            animation_options={
                                "frame": {"duration": self.PCP_TRANSITION_MS, "redraw": True},
                                "transition": {"duration": self.PCP_TRANSITION_MS},
                            },
                        ),
                        style={"width": "50%", "display": "inline-block"},
                    ),
                    html.Div(
                        dcc.Graph(
                            id=self._get_id(f"pcp-{self.services[2]}"),
                            animate=True,
                            animation_options={
                                "frame": {"duration": self.PCP_TRANSITION_MS, "redraw": True},
                                "transition": {"duration": self.PCP_TRANSITION_MS},
                            },
                        ),
                        style={"width": "50%", "display": "inline-block"},
                    ),
                    html.Div(
                        dcc.Graph(
                            id=self._get_id(f"pcp-{self.services[3]}"),
                            animate=True,
                            animation_options={
                                "frame": {"duration": self.PCP_TRANSITION_MS, "redraw": True},
                                "transition": {"duration": self.PCP_TRANSITION_MS},
                            },
                        ),
                        style={"width": "50%", "display": "inline-block"},
                    ),
                ]),
            ], style={"width": "100%", "display": "inline-block", "verticalAlign": "top"}),

        ], style=self.page_style)

    def register_callbacks(self, app):
        """
        Register callbacks with the Dash app.

        Args:
            app: The Dash app instance
        """

        @app.callback(
            Output(self._get_id("selected-sat-bin"), "data"),
            Input(self._get_id("satisfaction-hist"), "clickData"),
            State(self._get_id("selected-sat-bin"), "data"),
            prevent_initial_call=True,
        )
        def toggle_selected_bin(click_data, current_bin):
            """Toggle selected histogram bin on click."""
            if click_data is None or "points" not in click_data:
                return current_bin

            x_clicked = click_data["points"][0]["x"]  # bin center
            idx = int(np.argmin(np.abs(self.SAT_BIN_CENTERS - x_clicked)))

            if current_bin is not None and current_bin.get("bin_idx") == idx:
                return None

            return {
                "bin_idx": idx,
                "low": float(self.SAT_BINS[idx]),
                "high": float(self.SAT_BINS[idx + 1]),
            }

        @app.callback(
            Output(self._get_id("satisfaction-hist"), "figure"),
            [Input(self._get_id("week-range"), "value"),
             Input(self._get_id("selected-sat-bin"), "data")],
        )
        def update_histogram(week_range, selected_bin):
            """Update histogram of patient satisfaction for selected weeks."""
            return self._create_histogram(week_range, selected_bin)

        @app.callback(
            [
                Output(self._get_id(f"pcp-{self.services[0]}"), "figure"),
                Output(self._get_id(f"pcp-{self.services[1]}"), "figure"),
                Output(self._get_id(f"pcp-{self.services[2]}"), "figure"),
                Output(self._get_id(f"pcp-{self.services[3]}"), "figure"),
            ],
            [Input(self._get_id("week-range"), "value"),
             Input(self._get_id("selected-sat-bin"), "data")],
        )
        def update_all_pcps(week_range, selected_bin):
            """Update all PCP plots when week range or bin selection changes."""
            return self._update_all_pcps(week_range, selected_bin)

    def _create_histogram(self, week_range, selected_bin):
        """Create histogram of patient satisfaction for selected weeks."""
        w1, w2 = week_range
        dff = self.df[(self.df["week"] >= w1) & (self.df["week"] <= w2)]

        vals = dff["patient_satisfaction"].dropna().values
        counts, _ = np.histogram(vals, bins=self.SAT_BINS)

        norm_centers = (self.SAT_BIN_CENTERS - 0) / 100.0
        base_colors = sample_colorscale("Viridis", norm_centers.tolist())

        if selected_bin is not None:
            selected_idx = selected_bin["bin_idx"]
            colors = [
                base_colors[i] if i == selected_idx else "rgba(220,220,220,0.7)"
                for i in range(len(counts))
            ]
        else:
            colors = base_colors

        fig = go.Figure(
            data=[
                go.Bar(
                    x=self.SAT_BIN_CENTERS,
                    y=counts,
                    marker=dict(color=colors),
                    hovertemplate="Satisfaction: %{customdata}<br>Count: %{y}<extra></extra>",
                    customdata=[
                        f"{int(self.SAT_BINS[i])}–{int(self.SAT_BINS[i+1])}"
                        for i in range(len(counts))
                    ],
                )
            ]
        )

        fig.update_layout(
            title="Patient Satisfaction Distribution",
            xaxis_title="Satisfaction bin",
            yaxis_title="Count",
            template="plotly_white",
            height=260,
            margin=dict(t=40, l=40, r=20, b=60),
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=self.SAT_BIN_CENTERS,
            ticktext=[
                f"{int(self.SAT_BINS[i])}-{int(self.SAT_BINS[i+1])}"
                for i in range(len(self.SAT_BIN_CENTERS))
            ],
        )

        return fig

    def _update_all_pcps(self, week_range, selected_bin):
        """Update all PCP plots based on week range and selected bin."""
        w1, w2 = week_range

        #base data by week range (used by PCP ranges)
        dff_weeks = self.df[(self.df["week"] >= w1) & (self.df["week"] <= w2)].copy()

        #PCP data optionally filtered by histogram satisfaction bin
        dff_pcp = dff_weeks.copy()
        if selected_bin is not None:
            low, high = selected_bin["low"], selected_bin["high"]
            dff_pcp = dff_pcp[
                (dff_pcp["patient_satisfaction"] >= low)
                & (dff_pcp["patient_satisfaction"] < high)
            ]

        #PCP figures (one per service)
        pcp_figs = []
        for s in self.services:
            pcp_figs.append(self._create_pcp_figure(s, dff_weeks, dff_pcp))

        return pcp_figs

    def _create_pcp_figure(self, service, dff_weeks, dff_pcp):
        """Create a parallel coordinates plot for a single service."""
        
        #data for ranges: all weeks for this service
        dff_service_weeks = dff_weeks[dff_weeks["service"] == service].copy()


        #data for lines: filtered by bin (if any)
        dff = dff_pcp[dff_pcp["service"] == service].copy()

        needed_cols = ["patient_satisfaction"] + [c for _, c in self.PCP_DIMS]
        needed_cols = [c for c in needed_cols if c in dff.columns]
        dff = dff.dropna(subset=needed_cols, how="any")

        if dff.empty or dff_service_weeks.empty:
            return go.Figure(
                layout=dict(
                    title=f"No Data Available — {service}",
                    xaxis={"visible": False},
                    yaxis={"visible": False},
                    height=430,
                    template="plotly_white",
                    transition={"duration": self.PCP_TRANSITION_MS},
                    uirevision="keep",
                )
            )

        dimensions = []
        for label, col in self.PCP_DIMS:
            if col not in dff.columns:
                continue

            #ranges from full week-filtered service data
            vmin = float(np.nanmin(dff_service_weeks[col].values))
            vmax = float(np.nanmax(dff_service_weeks[col].values))

            is_ratio = "ratio" in col.lower()
            fmt = (lambda x: f"{x:.2f}") if is_ratio else (lambda x: f"{x:.0f}")

            dimensions.append(
                dict(
                    label=label,
                    values=dff[col],
                    range=[vmin, vmax],
                    tickvals=[vmin, vmax],
                    ticktext=[fmt(vmin), fmt(vmax)],
                )
            )

        sat = dff["patient_satisfaction"]
        cmin, cmax = 0, 100

        fig = go.Figure(
            data=[
                go.Parcoords(
                    labelside="bottom",
                    line=dict(
                        color=sat,
                        colorscale="Viridis",
                        cmin=cmin,
                        cmax=cmax,
                        showscale=True,
                        colorbar=dict(title="Patient Satisfaction"),
                    ),
                    dimensions=dimensions,
                )
            ]
        )

        fig.update_layout(
            title=f"Parallel Coordinates — {service}",
            template="plotly_white",
            height=430,
            margin=dict(t=60, l=30, r=30, b=70),
            transition={"duration": self.PCP_TRANSITION_MS},
            uirevision="keep",
        )

        return fig