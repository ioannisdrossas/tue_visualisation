
# # libraries
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.colors import sample_colorscale
# from dash import Dash, dcc, html, Input, Output, State

# # load data
# df = pd.read_csv("services_weekly.csv")
# staff = pd.read_csv("staff.csv")
# schedule = pd.read_csv("staff_schedule.csv")

# # feature engineering & prep

# # count nurses present per week and service
# nurse_counts = (
#     schedule[schedule["role"].str.lower() == "nurse"]
#     .groupby(["week", "service"])["present"]
#     .sum()
#     .reset_index(name="nurses_present")
# )

# # count doctors present per week and service
# doctor_counts = (
#     schedule[schedule["role"].str.lower() == "doctor"]
#     .groupby(["week", "service"])["present"]
#     .sum()
#     .reset_index(name="doctors_present")
# )

# # add staff counts to the main service dataset
# df = df.merge(nurse_counts, on=["week", "service"], how="left")
# df = df.merge(doctor_counts, on=["week", "service"], how="left")

# # replace zero staff counts to avoid division by zero
# df.loc[df["nurses_present"] == 0, "nurses_present"] = pd.NA
# df.loc[df["doctors_present"] == 0, "doctors_present"] = pd.NA

# # compute patient-to-staff ratios
# df["patient_to_nurse_ratio"] = df["patients_admitted"] / df["nurses_present"]
# df["patient_to_doctor_ratio"] = df["patients_admitted"] / df["doctors_present"]

# # select the first four services to display
# services = sorted(df["service"].unique())[:4]

# # define minimum and maximum weeks for the time range slider
# week_min = int(df["week"].min())
# week_max = int(df["week"].max())

# # histogram bin edges 
# SAT_BINS = np.arange(0, 110, 10)
# SAT_BIN_CENTERS = (SAT_BINS[:-1] + SAT_BINS[1:]) / 2.0

# # app
# app = Dash(__name__)

# # page-level styling
# PAGE_STYLE = {
#     "fontFamily": "Inter, Arial, sans-serif",
#     "padding": "20px",
#     "backgroundColor": "white",
#     "minHeight": "100vh",
# }

# # header styling
# HEADER_STYLE = {"marginBottom": "10px"}

# # scroll container styling for long charts
# SCROLL_WRAP_STYLE = {
#     "height": "750px",
#     "overflowY": "scroll",
#     "paddingRight": "15px",
#     "border": "1px solid #eee",
# }

# # exactly 4 attributes under the PCP
# PCP_DIMS = [
#     ("Staff Morale", "staff_morale"),
#     ("Available Beds", "available_beds"),
#     ("Patient-to-Nurse Ratio", "patient_to_nurse_ratio"),
#     ("Patient-to-Doctor Ratio", "patient_to_doctor_ratio"),
# ]

# # animation / transition settings (ms)
# PCP_TRANSITION_MS = 350
# STAFF_TRANSITION_MS = 350

# # layout
# app.layout = html.Div(
#     [
#         # store for selected satisfaction bin (for PCP only)
#         dcc.Store(id="selected-sat-bin", data=None),

#         html.H2("Service Metrics Dashboard", style=HEADER_STYLE),

#         # Week range slider
#         html.Div(
#             [
#                 html.Label("Time Range (weeks)"),
#                 dcc.RangeSlider(
#                     id="week-range",
#                     min=week_min,
#                     max=week_max,
#                     step=1,
#                     value=[week_min, week_max],
#                     allowCross=False,
#                     marks={w: str(w) for w in range(week_min, week_max + 1, 5)},
#                     updatemode="mouseup",
#                 ),
#             ],
#             style={"marginBottom": "16px"},
#         ),

#         # Histogram under week bar
#         html.Div(
#             [
#                 html.H4("Patient Satisfaction Histogram (Click a bin to filter PCP lines)"),
#                 dcc.Graph(id="satisfaction-hist"),
#                 html.Div(
#                     "Tip: click a bar to select a bin; click it again to clear.",
#                     style={"fontSize": "12px", "color": "#777"},
#                 ),
#             ],
#             style={"marginBottom": "20px"},
#         ),

#         html.Div(
#             [
#                 # PCP plots
#                 html.Div(
#                     [
#                         html.H4("Parallel Coordinates (Color: Patient Satisfaction)"),
#                         html.Div(
#                             [
#                                 html.Div(
#                                     dcc.Graph(
#                                         id=f"pcp-{services[0]}",
#                                         animate=True,
#                                         animation_options={
#                                             "frame": {
#                                                 "duration": PCP_TRANSITION_MS,
#                                                 "redraw": True,
#                                             },
#                                             "transition": {
#                                                 "duration": PCP_TRANSITION_MS
#                                             },
#                                         },
#                                     ),
#                                     style={"width": "50%", "display": "inline-block"},
#                                 ),
#                                 html.Div(
#                                     dcc.Graph(
#                                         id=f"pcp-{services[1]}",
#                                         animate=True,
#                                         animation_options={
#                                             "frame": {
#                                                 "duration": PCP_TRANSITION_MS,
#                                                 "redraw": True,
#                                             },
#                                             "transition": {
#                                                 "duration": PCP_TRANSITION_MS
#                                             },
#                                         },
#                                     ),
#                                     style={"width": "50%", "display": "inline-block"},
#                                 ),
#                                 html.Div(
#                                     dcc.Graph(
#                                         id=f"pcp-{services[2]}",
#                                         animate=True,
#                                         animation_options={
#                                             "frame": {
#                                                 "duration": PCP_TRANSITION_MS,
#                                                 "redraw": True,
#                                             },
#                                             "transition": {
#                                                 "duration": PCP_TRANSITION_MS
#                                             },
#                                         },
#                                     ),
#                                     style={"width": "50%", "display": "inline-block"},
#                                 ),
#                                 html.Div(
#                                     dcc.Graph(
#                                         id=f"pcp-{services[3]}",
#                                         animate=True,
#                                         animation_options={
#                                             "frame": {
#                                                 "duration": PCP_TRANSITION_MS,
#                                                 "redraw": True,
#                                             },
#                                             "transition": {
#                                                 "duration": PCP_TRANSITION_MS
#                                             },
#                                         },
#                                     ),
#                                     style={"width": "50%", "display": "inline-block"},
#                                 ),
#                             ]
#                         ),
#                     ],
#                     style={
#                         "width": "50%",
#                         "display": "inline-block",
#                         "verticalAlign": "top",
#                     },
#                 ),

#                 # Staff chart
#                 html.Div(
#                     [
#                         html.H4("Staff Association with Patient Satisfaction (Scrollable)"),
#                         html.Div(
#                             [
#                                 dcc.Graph(
#                                     id="staff-chart",
#                                     animate=True,
#                                     animation_options={
#                                         "frame": {
#                                             "duration": STAFF_TRANSITION_MS,
#                                             "redraw": True,
#                                         },
#                                         "transition": {
#                                             "duration": STAFF_TRANSITION_MS
#                                         },
#                                     },
#                                 )
#                             ],
#                             style=SCROLL_WRAP_STYLE,
#                         ),
#                     ],
#                     style={
#                         "width": "50%",
#                         "display": "inline-block",
#                         "verticalAlign": "top",
#                     },
#                 ),
#             ],
#             style={"display": "flex", "gap": "12px"},
#         ),
#     ],
#     style=PAGE_STYLE,
# )

# # ---------- Histogram callbacks ----------

# @app.callback(
#     Output("selected-sat-bin", "data"),
#     Input("satisfaction-hist", "clickData"),
#     State("selected-sat-bin", "data"),
#     prevent_initial_call=True,
# )
# def toggle_selected_bin(click_data, current_bin):
#     """Toggle selected histogram bin on click."""
#     if click_data is None or "points" not in click_data:
#         return current_bin

#     x_clicked = click_data["points"][0]["x"]  # bin center
#     idx = int(np.argmin(np.abs(SAT_BIN_CENTERS - x_clicked)))

#     if current_bin is not None and current_bin.get("bin_idx") == idx:
#         return None

#     return {
#         "bin_idx": idx,
#         "low": float(SAT_BINS[idx]),
#         "high": float(SAT_BINS[idx + 1]),
#     }


# @app.callback(
#     Output("satisfaction-hist", "figure"),
#     [Input("week-range", "value"), Input("selected-sat-bin", "data")],
# )
# def update_histogram(week_range, selected_bin):
#     """Update histogram of patient satisfaction for selected weeks."""
#     w1, w2 = week_range
#     dff = df[(df["week"] >= w1) & (df["week"] <= w2)]

#     vals = dff["patient_satisfaction"].dropna().values
#     counts, _ = np.histogram(vals, bins=SAT_BINS)

#     norm_centers = (SAT_BIN_CENTERS - 0) / 100.0
#     base_colors = sample_colorscale("Viridis", norm_centers.tolist())

#     if selected_bin is not None:
#         selected_idx = selected_bin["bin_idx"]
#         colors = [
#             base_colors[i] if i == selected_idx else "rgba(220,220,220,0.7)"
#             for i in range(len(counts))
#         ]
#     else:
#         colors = base_colors

#     fig = go.Figure(
#         data=[
#             go.Bar(
#                 x=SAT_BIN_CENTERS,
#                 y=counts,
#                 marker=dict(color=colors),
#                 hovertemplate="Satisfaction: %{customdata}<br>Count: %{y}<extra></extra>",
#                 customdata=[
#                     f"{int(SAT_BINS[i])}–{int(SAT_BINS[i+1])}"
#                     for i in range(len(counts))
#                 ],
#             )
#         ]
#     )

#     fig.update_layout(
#         title="Patient Satisfaction Distribution",
#         xaxis_title="Satisfaction bin",
#         yaxis_title="Count",
#         template="plotly_white",
#         height=260,
#         margin=dict(t=40, l=40, r=20, b=60),
#     )

#     fig.update_xaxes(
#         tickmode="array",
#         tickvals=SAT_BIN_CENTERS,
#         ticktext=[
#             f"{int(SAT_BINS[i])}-{int(SAT_BINS[i+1])}"
#             for i in range(len(SAT_BIN_CENTERS))
#         ],
#     )

#     return fig

# # ---------- PCP + staff callback ----------

# @app.callback(
#     [
#         Output(f"pcp-{services[0]}", "figure"),
#         Output(f"pcp-{services[1]}", "figure"),
#         Output(f"pcp-{services[2]}", "figure"),
#         Output(f"pcp-{services[3]}", "figure"),
#         Output("staff-chart", "figure"),
#     ],
#     [Input("week-range", "value"), Input("selected-sat-bin", "data")],
# )
# def update_all(week_range, selected_bin):
#     w1, w2 = week_range

#     # base data by week range (used by staff chart and PCP ranges)
#     dff_weeks = df[(df["week"] >= w1) & (df["week"] <= w2)].copy()

#     # PCP data optionally filtered by histogram satisfaction bin
#     dff_pcp = dff_weeks.copy()
#     if selected_bin is not None:
#         low, high = selected_bin["low"], selected_bin["high"]
#         dff_pcp = dff_pcp[
#             (dff_pcp["patient_satisfaction"] >= low)
#             & (dff_pcp["patient_satisfaction"] < high)
#         ]

#     # PCP figures (one per service)
#     pcp_figs = []
#     for s in services:
#         # data for ranges: all weeks for this service
#         dff_service_weeks = dff_weeks[dff_weeks["service"] == s].copy()

#         # data for lines: filtered by bin (if any)
#         dff = dff_pcp[dff_pcp["service"] == s].copy()

#         needed_cols = ["patient_satisfaction"] + [c for _, c in PCP_DIMS]
#         needed_cols = [c for c in needed_cols if c in dff.columns]
#         dff = dff.dropna(subset=needed_cols, how="any")

#         if dff.empty or dff_service_weeks.empty:
#             fig_empty = go.Figure(
#                 layout=dict(
#                     title=f"No Data Available — {s}",
#                     xaxis={"visible": False},
#                     yaxis={"visible": False},
#                     height=430,
#                     template="plotly_white",
#                     transition={"duration": PCP_TRANSITION_MS},
#                     uirevision="keep",
#                 )
#             )
#             pcp_figs.append(fig_empty)
#             continue

#         dimensions = []
#         for label, col in PCP_DIMS:
#             if col not in dff.columns:
#                 continue

#             # ranges from full week-filtered service data
#             vmin = float(np.nanmin(dff_service_weeks[col].values))
#             vmax = float(np.nanmax(dff_service_weeks[col].values))

#             is_ratio = "ratio" in col.lower()
#             fmt = (lambda x: f"{x:.2f}") if is_ratio else (lambda x: f"{x:.0f}")

#             dimensions.append(
#                 dict(
#                     label=label,
#                     values=dff[col],
#                     range=[vmin, vmax],
#                     tickvals=[vmin, vmax],
#                     ticktext=[fmt(vmin), fmt(vmax)],
#                 )
#             )

#         sat = dff["patient_satisfaction"]
#         cmin, cmax = 0, 100

#         fig = go.Figure(
#             data=[
#                 go.Parcoords(
#                     labelside="bottom",
#                     line=dict(
#                         color=sat,
#                         colorscale="Viridis",
#                         cmin=cmin,
#                         cmax=cmax,
#                         showscale=True,
#                         colorbar=dict(title="Patient Satisfaction"),
#                     ),
#                     dimensions=dimensions,
#                 )
#             ]
#         )

#         fig.update_layout(
#             title=f"Parallel Coordinates — {s}",
#             template="plotly_white",
#             height=430,
#             margin=dict(t=60, l=30, r=30, b=70),
#             transition={"duration": PCP_TRANSITION_MS},
#             uirevision="keep",
#         )

#         pcp_figs.append(fig)

#     # Staff chart uses only week filter (no satisfaction-bin filter)
#     if dff_weeks.empty:
#         staff_fig = go.Figure(
#             layout={
#                 "title": "No Data Available for Selected Week Range",
#                 "xaxis": {"visible": False},
#                 "yaxis": {"visible": False},
#                 "height": 700,
#                 "template": "plotly_white",
#                 "transition": {"duration": STAFF_TRANSITION_MS},
#                 "uirevision": "keep",
#             }
#         )
#         return pcp_figs + [staff_fig]

#     overall_avg_satisfaction = dff_weeks["patient_satisfaction"].mean()

#     sched_filtered = schedule[(schedule["week"] >= w1) & (schedule["week"] <= w2)]
#     staff_satisfaction = (
#         sched_filtered.merge(
#             dff_weeks[["week", "service", "patient_satisfaction"]],
#             on=["week", "service"],
#             how="left",
#         )
#         .merge(staff, on="staff_id", how="left")
#     )

#     name_col_candidates = ["staff_name_x", "staff_name", "staff_name_y", "name"]
#     name_col = next(
#         (c for c in name_col_candidates if c in staff_satisfaction.columns), "staff_id"
#     )

#     staff_agg = (
#         staff_satisfaction.groupby(["staff_id", name_col], as_index=False)
#         .agg(mean_satisfaction=("patient_satisfaction", "mean"))
#         .dropna(subset=["mean_satisfaction"])
#     )

#     staff_agg["satisfaction_contribution"] = (
#         staff_agg["mean_satisfaction"] - overall_avg_satisfaction
#     )
#     staff_agg = staff_agg.sort_values("satisfaction_contribution", ascending=True)

#     bar_colors = [
#         "#27AE60" if v >= 0 else "#E74C3C"
#         for v in staff_agg["satisfaction_contribution"]
#     ]

#     staff_fig = go.Figure()
#     staff_fig.add_trace(
#         go.Bar(
#             x=staff_agg["satisfaction_contribution"],
#             y=staff_agg[name_col],
#             orientation="h",
#             marker_color=bar_colors,
#             text=[f"{v:+.2f}" for v in staff_agg["satisfaction_contribution"]],
#             textposition="auto",
#         )
#     )
#     staff_fig.update_layout(
#         title={
#             "text": f"Staff Association with Patient Satisfaction Deviation (Weeks {w1}–{w2})",
#             "x": 0.5,
#         },
#         xaxis_title=f"Avg Satisfaction Deviation from Overall Mean ({overall_avg_satisfaction:.2f})",
#         yaxis_title="Staff Member",
#         height=2500,
#         template="plotly_white",
#         margin=dict(t=80, l=60, r=60, b=60),
#         transition={"duration": STAFF_TRANSITION_MS},
#         uirevision="keep",
#         shapes=[
#             dict(
#                 type="line",
#                 x0=0,
#                 x1=0,
#                 y0=-0.5,
#                 y1=max(0, len(staff_agg[name_col]) - 0.5),
#                 line=dict(color="#2c3e50", width=2),
#             )
#         ],
#     )

#     return pcp_figs + [staff_fig]


# # run
# # http://127.0.0.1:8050
# if __name__ == "__main__":
#     app.run(debug=True)


# ########################################



# libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from dash import Dash, dcc, html, Input, Output, State

# load data
df = pd.read_csv("services_weekly.csv")
schedule = pd.read_csv("staff_schedule.csv")

# feature engineering & prep

# count nurses present per week and service
nurse_counts = (
    schedule[schedule["role"].str.lower() == "nurse"]
    .groupby(["week", "service"])["present"]
    .sum()
    .reset_index(name="nurses_present")
)

# count doctors present per week and service
doctor_counts = (
    schedule[schedule["role"].str.lower() == "doctor"]
    .groupby(["week", "service"])["present"]
    .sum()
    .reset_index(name="doctors_present")
)

# add staff counts to the main service dataset
df = df.merge(nurse_counts, on=["week", "service"], how="left")
df = df.merge(doctor_counts, on=["week", "service"], how="left")

# replace zero staff counts to avoid division by zero
df.loc[df["nurses_present"] == 0, "nurses_present"] = pd.NA
df.loc[df["doctors_present"] == 0, "doctors_present"] = pd.NA

# compute patient-to-staff ratios
df["patient_to_nurse_ratio"] = df["patients_admitted"] / df["nurses_present"]
df["patient_to_doctor_ratio"] = df["patients_admitted"] / df["doctors_present"]

# select the first four services to display
services = sorted(df["service"].unique())[:4]

# define minimum and maximum weeks for the time range slider
week_min = int(df["week"].min())
week_max = int(df["week"].max())

# histogram bin edges 
SAT_BINS = np.arange(0, 110, 10)
SAT_BIN_CENTERS = (SAT_BINS[:-1] + SAT_BINS[1:]) / 2.0

# app
app = Dash(__name__)

# page-level styling
PAGE_STYLE = {
    "fontFamily": "Inter, Arial, sans-serif",
    "padding": "20px",
    "backgroundColor": "white",
    "minHeight": "100vh",
}

# header styling
HEADER_STYLE = {"marginBottom": "10px"}

# exactly 4 attributes under the PCP
PCP_DIMS = [
    ("Staff Morale", "staff_morale"),
    ("Available Beds", "available_beds"),
    ("Patient-to-Nurse Ratio", "patient_to_nurse_ratio"),
    ("Patient-to-Doctor Ratio", "patient_to_doctor_ratio"),
]

# animation / transition settings (ms)
PCP_TRANSITION_MS = 350

# layout
app.layout = html.Div(
    [
        # store for selected satisfaction bin (for PCP only)
        dcc.Store(id="selected-sat-bin", data=None),

        html.H2("Service Metrics Dashboard", style=HEADER_STYLE),

        # Week range slider
        html.Div(
            [
                html.Label("Time Range (weeks)"),
                dcc.RangeSlider(
                    id="week-range",
                    min=week_min,
                    max=week_max,
                    step=1,
                    value=[week_min, week_max],
                    allowCross=False,
                    marks={w: str(w) for w in range(week_min, week_max + 1, 5)},
                    updatemode="mouseup",
                ),
            ],
            style={"marginBottom": "16px"},
        ),

        # Histogram under week bar
        html.Div(
            [
                html.H4("Patient Satisfaction Histogram (Click a bin to filter PCP lines)"),
                dcc.Graph(id="satisfaction-hist"),
                html.Div(
                    "Tip: click a bar to select a bin; click it again to clear.",
                    style={"fontSize": "12px", "color": "#777"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),

        # PCP plots full width
        html.Div(
            [
                html.H4("Parallel Coordinates (Color: Patient Satisfaction)"),
                html.Div(
                    [
                        html.Div(
                            dcc.Graph(
                                id=f"pcp-{services[0]}",
                                animate=True,
                                animation_options={
                                    "frame": {
                                        "duration": PCP_TRANSITION_MS,
                                        "redraw": True,
                                    },
                                    "transition": {
                                        "duration": PCP_TRANSITION_MS
                                    },
                                },
                            ),
                            style={"width": "50%", "display": "inline-block"},
                        ),
                        html.Div(
                            dcc.Graph(
                                id=f"pcp-{services[1]}",
                                animate=True,
                                animation_options={
                                    "frame": {
                                        "duration": PCP_TRANSITION_MS,
                                        "redraw": True,
                                    },
                                    "transition": {
                                        "duration": PCP_TRANSITION_MS
                                    },
                                },
                            ),
                            style={"width": "50%", "display": "inline-block"},
                        ),
                        html.Div(
                            dcc.Graph(
                                id=f"pcp-{services[2]}",
                                animate=True,
                                animation_options={
                                    "frame": {
                                        "duration": PCP_TRANSITION_MS,
                                        "redraw": True,
                                    },
                                    "transition": {
                                        "duration": PCP_TRANSITION_MS
                                    },
                                },
                            ),
                            style={"width": "50%", "display": "inline-block"},
                        ),
                        html.Div(
                            dcc.Graph(
                                id=f"pcp-{services[3]}",
                                animate=True,
                                animation_options={
                                    "frame": {
                                        "duration": PCP_TRANSITION_MS,
                                        "redraw": True,
                                    },
                                    "transition": {
                                        "duration": PCP_TRANSITION_MS
                                    },
                                },
                            ),
                            style={"width": "50%", "display": "inline-block"},
                        ),
                    ]
                ),
            ],
            style={
                "width": "100%",
                "display": "inline-block",
                "verticalAlign": "top",
            },
        ),
    ],
    style=PAGE_STYLE,
)

# ---------- Histogram callbacks ----------

@app.callback(
    Output("selected-sat-bin", "data"),
    Input("satisfaction-hist", "clickData"),
    State("selected-sat-bin", "data"),
    prevent_initial_call=True,
)
def toggle_selected_bin(click_data, current_bin):
    """Toggle selected histogram bin on click."""
    if click_data is None or "points" not in click_data:
        return current_bin

    x_clicked = click_data["points"][0]["x"]  # bin center
    idx = int(np.argmin(np.abs(SAT_BIN_CENTERS - x_clicked)))

    if current_bin is not None and current_bin.get("bin_idx") == idx:
        return None

    return {
        "bin_idx": idx,
        "low": float(SAT_BINS[idx]),
        "high": float(SAT_BINS[idx + 1]),
    }


@app.callback(
    Output("satisfaction-hist", "figure"),
    [Input("week-range", "value"), Input("selected-sat-bin", "data")],
)
def update_histogram(week_range, selected_bin):
    """Update histogram of patient satisfaction for selected weeks."""
    w1, w2 = week_range
    dff = df[(df["week"] >= w1) & (df["week"] <= w2)]

    vals = dff["patient_satisfaction"].dropna().values
    counts, _ = np.histogram(vals, bins=SAT_BINS)

    norm_centers = (SAT_BIN_CENTERS - 0) / 100.0
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
                x=SAT_BIN_CENTERS,
                y=counts,
                marker=dict(color=colors),
                hovertemplate="Satisfaction: %{customdata}<br>Count: %{y}<extra></extra>",
                customdata=[
                    f"{int(SAT_BINS[i])}–{int(SAT_BINS[i+1])}"
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
        tickvals=SAT_BIN_CENTERS,
        ticktext=[
            f"{int(SAT_BINS[i])}-{int(SAT_BINS[i+1])}"
            for i in range(len(SAT_BIN_CENTERS))
        ],
    )

    return fig

# ---------- PCP callback ----------

@app.callback(
    [
        Output(f"pcp-{services[0]}", "figure"),
        Output(f"pcp-{services[1]}", "figure"),
        Output(f"pcp-{services[2]}", "figure"),
        Output(f"pcp-{services[3]}", "figure"),
    ],
    [Input("week-range", "value"), Input("selected-sat-bin", "data")],
)
def update_all(week_range, selected_bin):
    w1, w2 = week_range

    # base data by week range (used by PCP ranges)
    dff_weeks = df[(df["week"] >= w1) & (df["week"] <= w2)].copy()

    # PCP data optionally filtered by histogram satisfaction bin
    dff_pcp = dff_weeks.copy()
    if selected_bin is not None:
        low, high = selected_bin["low"], selected_bin["high"]
        dff_pcp = dff_pcp[
            (dff_pcp["patient_satisfaction"] >= low)
            & (dff_pcp["patient_satisfaction"] < high)
        ]

    # PCP figures (one per service)
    pcp_figs = []
    for s in services:
        # data for ranges: all weeks for this service
        dff_service_weeks = dff_weeks[dff_weeks["service"] == s].copy()

        # data for lines: filtered by bin (if any)
        dff = dff_pcp[dff_pcp["service"] == s].copy()

        needed_cols = ["patient_satisfaction"] + [c for _, c in PCP_DIMS]
        needed_cols = [c for c in needed_cols if c in dff.columns]
        dff = dff.dropna(subset=needed_cols, how="any")

        if dff.empty or dff_service_weeks.empty:
            fig_empty = go.Figure(
                layout=dict(
                    title=f"No Data Available — {s}",
                    xaxis={"visible": False},
                    yaxis={"visible": False},
                    height=430,
                    template="plotly_white",
                    transition={"duration": PCP_TRANSITION_MS},
                    uirevision="keep",
                )
            )
            pcp_figs.append(fig_empty)
            continue

        dimensions = []
        for label, col in PCP_DIMS:
            if col not in dff.columns:
                continue

            # ranges from full week-filtered service data
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
            title=f"Parallel Coordinates — {s}",
            template="plotly_white",
            height=430,
            margin=dict(t=60, l=30, r=30, b=70),
            transition={"duration": PCP_TRANSITION_MS},
            uirevision="keep",
        )

        pcp_figs.append(fig)

    return pcp_figs


# run
# http://127.0.0.1:8050
if __name__ == "__main__":
    app.run(debug=True)
