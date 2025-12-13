#libraries

import pandas as pd
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

#load data

df = pd.read_csv("services_weekly.csv")
staff = pd.read_csv("staff.csv")
schedule = pd.read_csv("staff_schedule.csv")


#feature engineering & prep

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


#add staff counts to the main service dataset
df = df.merge(nurse_counts, on=["week", "service"], how="left")
df = df.merge(doctor_counts, on=["week", "service"], how="left")


#replace zero staff counts to avoid division by zero
df.loc[df["nurses_present"] == 0, "nurses_present"] = pd.NA
df.loc[df["doctors_present"] == 0, "doctors_present"] = pd.NA


#compute patient-to-staff ratios
df["patient_to_nurse_ratio"] = df["patients_admitted"] / df["nurses_present"]
df["patient_to_doctor_ratio"] = df["patients_admitted"] / df["doctors_present"]


#convert satisfaction score from 0–100 to 1–5 scale
df["satisfaction_5pt"] = 1 + 4 * (df["patient_satisfaction"] / 100)


#controls

#resource metrics available for selection in the dropdown
metrics = {
    "Staff Morale": "staff_morale",
    "Available Beds": "available_beds",
    "Patient-to-Nurse Ratio": "patient_to_nurse_ratio",
    "Patient-to-Doctor Ratio": "patient_to_doctor_ratio",
}


#select the first four services to display in the scatter plots
services = sorted(df["service"].unique())[:4]


#define minimum and maximum weeks for the time range slider
week_min = int(df["week"].min())
week_max = int(df["week"].max())

#app

app = Dash(__name__)


#page-level styling
PAGE_STYLE = {
    "fontFamily": "Inter, Arial, sans-serif",
    "padding": "20px",
    "backgroundColor": "white",
    "minHeight": "100vh",
}

#header styling
HEADER_STYLE = {"marginBottom": "10px"}


#scroll container styling for long charts
SCROLL_WRAP_STYLE = {
    "height": "750px",
    "overflowY": "scroll",
    "paddingRight": "15px",
    "border": "1px solid #eee",
}


#define app layout
app.layout = html.Div(
    [
        #dashboard title
        html.H2("Service Metrics Dashboard", style=HEADER_STYLE),

        html.Div(
            [
                html.Label("Resource metric"),
                dcc.Dropdown(
                    id="metric-dd",
                    options=[{"label": k, "value": v} for k, v in metrics.items()],
                    value=list(metrics.values())[0],
                    clearable=False,
                    style={"width": "320px"},
                ),
                html.Br(),
                html.Label("Time Range (weeks)"),
                dcc.RangeSlider(
                    id="week-range",
                    min=week_min,
                    max=week_max,
                    step=1,
                    value=[week_min, week_max],
                    allowCross=False,
                    marks={w: str(w) for w in range(week_min, week_max + 1, 5)},
                ),
            ],
            style={"marginBottom": "16px"},
        ),

        #scatter plots (left) and staff chart (right)
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Resource vs Patient Satisfaction (Scatter Plots)"),
                        html.Div(
                            [
                                html.Div(dcc.Graph(id=f"scatter-{services[0]}"),
                                         style={"width": "50%", "display": "inline-block"}),
                                html.Div(dcc.Graph(id=f"scatter-{services[1]}"),
                                         style={"width": "50%", "display": "inline-block"}),
                                html.Div(dcc.Graph(id=f"scatter-{services[2]}"),
                                         style={"width": "50%", "display": "inline-block"}),
                                html.Div(dcc.Graph(id=f"scatter-{services[3]}"),
                                         style={"width": "50%", "display": "inline-block"}),
                            ]
                        ),
                    ],
                    style={"width": "50%", "display": "inline-block", "verticalAlign": "top"},
                ),
                html.Div(
                    [
                        html.H4("Staff Association with Patient Satisfaction (Scrollable)"),
                        html.Div([dcc.Graph(id="staff-chart")], style=SCROLL_WRAP_STYLE),
                    ],
                    style={"width": "50%", "display": "inline-block", "verticalAlign": "top"},
                ),
            ],
            style={"display": "flex", "gap": "12px"},
        ),
    ],
    style=PAGE_STYLE,
)


#callback


#update all charts when the metric or week range changes
@app.callback(
    [
        Output(f"scatter-{services[0]}", "figure"),
        Output(f"scatter-{services[1]}", "figure"),
        Output(f"scatter-{services[2]}", "figure"),
        Output(f"scatter-{services[3]}", "figure"),
        Output("staff-chart", "figure"),
    ],
    [
        Input("metric-dd", "value"),
        Input("week-range", "value"),
    ],
)


def update_all(selected_metric, week_range):
    inv_metrics = {v: k for k, v in metrics.items()}
    w1, w2 = week_range

    dff_service = df[(df["week"] >= w1) & (df["week"] <= w2)]


    #build scatterplots
    scatter_figs = []
    for s in services:
        dff = dff_service[dff_service["service"] == s]
        fig_sc = px.scatter(
            dff,
            x="patient_satisfaction",
            y=selected_metric,
            trendline="ols",
            hover_data=["week"],
            labels={
                "patient_satisfaction": "Patient Satisfaction",
                selected_metric: inv_metrics[selected_metric],
            },
            title=f"{inv_metrics[selected_metric]} vs Patient Satisfaction — {s}",
            template="plotly_white",
        )
        scatter_figs.append(fig_sc)


    #if no data in this week range, return empty staff chart
    if dff_service.empty:
        staff_fig = go.Figure(
            layout={
                "title": "No Data Available for Selected Week Range",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "height": 700,
            }
        )
        return scatter_figs + [staff_fig]


    #compute overall average satisfaction for baseline comparison
    overall_avg_satisfaction = dff_service["satisfaction_5pt"].mean()


    #filter staff schedule to selected weeks
    sched_filtered = schedule[(schedule["week"] >= w1) & (schedule["week"] <= w2)]

    staff_satisfaction = sched_filtered.merge(
        dff_service[["week", "service", "satisfaction_5pt"]],
        on=["week", "service"],
        how="left",
    ).merge(staff, on="staff_id", how="left")


    #pick the best available staff name column
    name_col_candidates = ["staff_name_x", "staff_name", "staff_name_y", "name"]
    name_col = next((c for c in name_col_candidates if c in staff_satisfaction.columns), "staff_id")


    #average satisfaction linked to each staff member
    staff_agg = (
        staff_satisfaction.groupby(["staff_id", name_col], as_index=False)
        .agg(mean_satisfaction=("satisfaction_5pt", "mean"))
        .dropna(subset=["mean_satisfaction"])
    )
    
    
    #compute deviation from the overall average satisfaction
    staff_agg["satisfaction_contribution"] = staff_agg["mean_satisfaction"] - overall_avg_satisfaction
    staff_agg = staff_agg.sort_values("satisfaction_contribution", ascending=True)

    bar_colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in staff_agg["satisfaction_contribution"]]

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
        title={"text": f"Staff Association with Patient Satisfaction Deviation (Weeks {w1}–{w2})", "x": 0.5},
        xaxis_title=f"Avg Satisfaction Deviation from Overall Mean ({overall_avg_satisfaction:.2f})",
        yaxis_title="Staff Member",
        height=2500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=80, l=60, r=60, b=60),
        shapes=[
            dict(
                type="line",
                x0=0, x1=0,
                y0=-0.5,
                y1=max(0, len(staff_agg[name_col]) - 0.5),
                line=dict(color="#2c3e50", width=2),
            )
        ],
    )

    return scatter_figs + [staff_fig]

#run
#http://127.0.0.1:8050


if __name__ == "__main__":
    app.run(debug=True)
