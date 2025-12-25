# # libraries
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from dash import Dash, dcc, html, Input, Output

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
#         html.H2("Service Metrics Dashboard", style=HEADER_STYLE),

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
#                     # Optional: reduces “spam updates” while dragging.
#                     # With this, callback fires when you release the slider.
#                     updatemode="mouseup",
#                 ),
#             ],
#             style={"marginBottom": "16px"},
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
#                                         animation_options={"frame": {"duration": PCP_TRANSITION_MS, "redraw": True},
#                                                            "transition": {"duration": PCP_TRANSITION_MS}},
#                                     ),
#                                     style={"width": "50%", "display": "inline-block"},
#                                 ),
#                                 html.Div(
#                                     dcc.Graph(
#                                         id=f"pcp-{services[1]}",
#                                         animate=True,
#                                         animation_options={"frame": {"duration": PCP_TRANSITION_MS, "redraw": True},
#                                                            "transition": {"duration": PCP_TRANSITION_MS}},
#                                     ),
#                                     style={"width": "50%", "display": "inline-block"},
#                                 ),
#                                 html.Div(
#                                     dcc.Graph(
#                                         id=f"pcp-{services[2]}",
#                                         animate=True,
#                                         animation_options={"frame": {"duration": PCP_TRANSITION_MS, "redraw": True},
#                                                            "transition": {"duration": PCP_TRANSITION_MS}},
#                                     ),
#                                     style={"width": "50%", "display": "inline-block"},
#                                 ),
#                                 html.Div(
#                                     dcc.Graph(
#                                         id=f"pcp-{services[3]}",
#                                         animate=True,
#                                         animation_options={"frame": {"duration": PCP_TRANSITION_MS, "redraw": True},
#                                                            "transition": {"duration": PCP_TRANSITION_MS}},
#                                     ),
#                                     style={"width": "50%", "display": "inline-block"},
#                                 ),
#                             ]
#                         ),
#                     ],
#                     style={"width": "50%", "display": "inline-block", "verticalAlign": "top"},
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
#                                     animation_options={"frame": {"duration": STAFF_TRANSITION_MS, "redraw": True},
#                                                        "transition": {"duration": STAFF_TRANSITION_MS}},
#                                 )
#                             ],
#                             style=SCROLL_WRAP_STYLE,
#                         ),
#                     ],
#                     style={"width": "50%", "display": "inline-block", "verticalAlign": "top"},
#                 ),
#             ],
#             style={"display": "flex", "gap": "12px"},
#         ),
#     ],
#     style=PAGE_STYLE,
# )

# # callback
# @app.callback(
#     [
#         Output(f"pcp-{services[0]}", "figure"),
#         Output(f"pcp-{services[1]}", "figure"),
#         Output(f"pcp-{services[2]}", "figure"),
#         Output(f"pcp-{services[3]}", "figure"),
#         Output("staff-chart", "figure"),
#     ],
#     [Input("week-range", "value")],
# )
# def update_all(week_range):
#     w1, w2 = week_range
#     dff_all = df[(df["week"] >= w1) & (df["week"] <= w2)].copy()

#     # PCP figures (one per service)
#     pcp_figs = []
#     for s in services:
#         dff = dff_all[dff_all["service"] == s].copy()

#         # Need satisfaction for color + the 4 dimensions
#         needed_cols = ["patient_satisfaction"] + [c for _, c in PCP_DIMS]
#         needed_cols = [c for c in needed_cols if c in dff.columns]

#         dff = dff.dropna(subset=needed_cols, how="any")

#         if dff.empty:
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

#         # Build dimensions (4 attributes)
#         dimensions = []
#         for label, col in PCP_DIMS:
#             if col not in dff.columns:
#                 continue

#             vmin = float(np.nanmin(dff[col].values))
#             vmax = float(np.nanmax(dff[col].values))

#             # format: ratios get 2 decimals, others as integers
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

#     # Staff chart logic
#     if dff_all.empty:
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

#     overall_avg_satisfaction = dff_all["patient_satisfaction"].mean()

#     sched_filtered = schedule[(schedule["week"] >= w1) & (schedule["week"] <= w2)]
#     staff_satisfaction = (
#         sched_filtered.merge(
#             dff_all[["week", "service", "patient_satisfaction"]],
#             on=["week", "service"],
#             how="left",
#         )
#         .merge(staff, on="staff_id", how="left")
#     )

#     name_col_candidates = ["staff_name_x", "staff_name", "staff_name_y", "name"]
#     name_col = next((c for c in name_col_candidates if c in staff_satisfaction.columns), "staff_id")

#     staff_agg = (
#         staff_satisfaction.groupby(["staff_id", name_col], as_index=False)
#         .agg(mean_satisfaction=("patient_satisfaction", "mean"))
#         .dropna(subset=["mean_satisfaction"])
#     )

#     staff_agg["satisfaction_contribution"] = staff_agg["mean_satisfaction"] - overall_avg_satisfaction
#     staff_agg = staff_agg.sort_values("satisfaction_contribution", ascending=True)

#     bar_colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in staff_agg["satisfaction_contribution"]]

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
#         title={"text": f"Staff Association with Patient Satisfaction Deviation (Weeks {w1}–{w2})", "x": 0.5},
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



 
# # For each staff member, we estimate satisfaction by averaging the patient satisfaction scores of the services they worked in during the selected weeks.
# # ------

# # 1) Each service–week has a patient satisfaction score.

# # 2) Each staff shift is linked to a service and a week.

# # 3) We attach the service’s satisfaction score to every shift worked by that staff member.

# # 4) For each staff member, we take the mean of all satisfaction scores from their shifts.

# # 5) We compare that mean to the overall average satisfaction for the same weeks.

# # 6) The difference is shown as the staff member’s satisfaction deviation (positive or negative).


# libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# load data
df = pd.read_csv("services_weekly.csv")
staff = pd.read_csv("staff.csv")
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

# scroll container styling for long charts
SCROLL_WRAP_STYLE = {
    "height": "750px",
    "overflowY": "scroll",
    "paddingRight": "15px",
    "border": "1px solid #eee",
}

# exactly 4 attributes under the PCP
PCP_DIMS = [
    ("Staff Morale", "staff_morale"),
    ("Available Beds", "available_beds"),
    ("Patient-to-Nurse Ratio", "patient_to_nurse_ratio"),
    ("Patient-to-Doctor Ratio", "patient_to_doctor_ratio"),
]

# animation / transition settings (ms)
PCP_TRANSITION_MS = 350
STAFF_TRANSITION_MS = 350

# layout
app.layout = html.Div(
    [
        html.H2("Service Metrics Dashboard", style=HEADER_STYLE),

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

        html.Div(
            [
                # PCP plots
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
                                            "transition": {"duration": PCP_TRANSITION_MS},
                                        },
                                    ),
                                    style={
                                        "width": "50%",
                                        "display": "inline-block",
                                    },
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
                                            "transition": {"duration": PCP_TRANSITION_MS},
                                        },
                                    ),
                                    style={
                                        "width": "50%",
                                        "display": "inline-block",
                                    },
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
                                            "transition": {"duration": PCP_TRANSITION_MS},
                                        },
                                    ),
                                    style={
                                        "width": "50%",
                                        "display": "inline-block",
                                    },
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
                                            "transition": {"duration": PCP_TRANSITION_MS},
                                        },
                                    ),
                                    style={
                                        "width": "50%",
                                        "display": "inline-block",
                                    },
                                ),
                            ]
                        ),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),

                # Staff chart
                html.Div(
                    [
                        html.H4("Staff Association with Patient Satisfaction (Scrollable)"),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="staff-chart",
                                    animate=True,
                                    animation_options={
                                        "frame": {
                                            "duration": STAFF_TRANSITION_MS,
                                            "redraw": True,
                                        },
                                        "transition": {"duration": STAFF_TRANSITION_MS},
                                    },
                                )
                            ],
                            style=SCROLL_WRAP_STYLE,
                        ),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ],
            style={"display": "flex", "gap": "12px"},
        ),

        # NEW: Histogram of patient satisfaction
        html.Div(
            [
                html.H4("Patient Satisfaction Distribution (Selected Weeks)"),
                dcc.Graph(
                    id="satisfaction-hist",
                    animate=True,
                    animation_options={
                        "frame": {"duration": STAFF_TRANSITION_MS, "redraw": True},
                        "transition": {"duration": STAFF_TRANSITION_MS},
                    },
                ),
            ],
            style={"marginTop": "24px"},
        ),
    ],
    style=PAGE_STYLE,
)

# callback
@app.callback(
    [
        Output(f"pcp-{services[0]}", "figure"),
        Output(f"pcp-{services[1]}", "figure"),
        Output(f"pcp-{services[2]}", "figure"),
        Output(f"pcp-{services[3]}", "figure"),
        Output("staff-chart", "figure"),
        Output("satisfaction-hist", "figure"),  # NEW output
    ],
    [Input("week-range", "value")],
)
def update_all(week_range):
    w1, w2 = week_range
    dff_all = df[(df["week"] >= w1) & (df["week"] <= w2)].copy()

    # PCP figures (one per service)
    pcp_figs = []
    for s in services:
        dff = dff_all[dff_all["service"] == s].copy()

        # Need satisfaction for color + the 4 dimensions
        needed_cols = ["patient_satisfaction"] + [c for _, c in PCP_DIMS]
        needed_cols = [c for c in needed_cols if c in dff.columns]

        dff = dff.dropna(subset=needed_cols, how="any")

        if dff.empty:
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

        # Build dimensions (4 attributes)
        dimensions = []
        for label, col in PCP_DIMS:
            if col not in dff.columns:
                continue

            vmin = float(np.nanmin(dff[col].values))
            vmax = float(np.nanmax(dff[col].values))

            # format: ratios get 2 decimals, others as integers
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

    # If no data at all for selected weeks
    if dff_all.empty:
        staff_fig = go.Figure(
            layout={
                "title": "No Data Available for Selected Week Range",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "height": 700,
                "template": "plotly_white",
                "transition": {"duration": STAFF_TRANSITION_MS},
                "uirevision": "keep",
            }
        )

        hist_fig = go.Figure(
            layout={
                "title": "No Data Available for Patient Satisfaction Histogram",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "height": 400,
                "template": "plotly_white",
                "transition": {"duration": STAFF_TRANSITION_MS},
                "uirevision": "keep",
            }
        )

        return pcp_figs + [staff_fig, hist_fig]

    # Staff chart logic
    overall_avg_satisfaction = dff_all["patient_satisfaction"].mean()

    sched_filtered = schedule[(schedule["week"] >= w1) & (schedule["week"] <= w2)]
    staff_satisfaction = (
        sched_filtered.merge(
            dff_all[["week", "service", "patient_satisfaction"]],
            on=["week", "service"],
            how="left",
        )
        .merge(staff, on="staff_id", how="left")
    )

    name_col_candidates = ["staff_name_x", "staff_name", "staff_name_y", "name"]
    name_col = next(
        (c for c in name_col_candidates if c in staff_satisfaction.columns),
        "staff_id",
    )

    staff_agg = (
        staff_satisfaction.groupby(["staff_id", name_col], as_index=False)
        .agg(mean_satisfaction=("patient_satisfaction", "mean"))
        .dropna(subset=["mean_satisfaction"])
    )

    staff_agg["satisfaction_contribution"] = (
        staff_agg["mean_satisfaction"] - overall_avg_satisfaction
    )
    staff_agg = staff_agg.sort_values("satisfaction_contribution", ascending=True)

    bar_colors = [
        "#27AE60" if v >= 0 else "#E74C3C"
        for v in staff_agg["satisfaction_contribution"]
    ]

    staff_fig = go.Figure()
    staff_fig.add_trace(
        go.Bar(
            x=staff_agg["satisfaction_contribution"],
            y=staff_agg[name_col],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+.2f}" for v in staff_agg["satisfaction_contribution"]],
            textposition="auto",
        )
    )
    staff_fig.update_layout(
        title={
            "text": f"Staff Association with Patient Satisfaction Deviation (Weeks {w1}–{w2})",
            "x": 0.5,
        },
        xaxis_title=f"Avg Satisfaction Deviation from Overall Mean ({overall_avg_satisfaction:.2f})",
        yaxis_title="Staff Member",
        height=2500,
        template="plotly_white",
        margin=dict(t=80, l=60, r=60, b=60),
        transition={"duration": STAFF_TRANSITION_MS},
        uirevision="keep",
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=0,
                y0=-0.5,
                y1=max(0, len(staff_agg[name_col]) - 0.5),
                line=dict(color="#2c3e50", width=2),
            )
        ],
    )

    # Histogram for patient satisfaction (all services combined for selected weeks)
    ps = dff_all["patient_satisfaction"].dropna()

    if ps.empty:
        hist_fig = go.Figure(
            layout={
                "title": "No Patient Satisfaction Data for Histogram (Selected Weeks)",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "height": 400,
                "template": "plotly_white",
                "transition": {"duration": STAFF_TRANSITION_MS},
                "uirevision": "keep",
            }
        )
    else:
        ps_min = float(ps.min())
        ps_max = float(ps.max())

        # Avoid zero-size bins if all values are equal
        if ps_min == ps_max:
            xbins = None
        else:
            bin_size = (ps_max - ps_min) / 10.0
            xbins = dict(start=ps_min, end=ps_max, size=bin_size)

        hist_trace_kwargs = {"x": ps}
        if xbins is not None:
            hist_trace_kwargs["xbins"] = xbins

        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Histogram(
                **hist_trace_kwargs,
            )
        )
        hist_fig.update_layout(
            title="Distribution of Patient Satisfaction (Selected Weeks)",
            xaxis_title="Patient Satisfaction",
            yaxis_title="Count of Records",
            template="plotly_white",
            height=400,
            margin=dict(t=60, l=60, r=40, b=60),
            transition={"duration": STAFF_TRANSITION_MS},
            uirevision="keep",
        )

    return pcp_figs + [staff_fig, hist_fig]


# run
# http://127.0.0.1:8050
if __name__ == "__main__":
    app.run(debug=True)
