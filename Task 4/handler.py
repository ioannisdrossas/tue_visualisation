if __name__ == "__main__":
    from dash import Dash, dcc, html
    import dash_bootstrap_components as dbc
    from task5 import ServicePerformanceDashboard
    from task23 import StaffingAnalysisDashboard
    from task4 import StaffMetricsDashboard

    # Create the Dash app
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Tab styles
    tab_style = {
        'borderRadius': '8px',
        'backgroundColor': '#2c3e50',
        'color': 'white',
        'padding': '10px 20px',
        'border': 'none',
        'margin': '0 4px'
    }

    tab_selected_style = {
        'borderRadius': '8px',
        'backgroundColor': '#1a1a1a',
        'color': 'white',
        'padding': '10px 20px',
        'border': 'none',
        'fontWeight': 'bold',
        'margin': '0 4px'
    }

    # Create 4 dashboard instances with unique ID prefixes
    dashboard1 = ServicePerformanceDashboard("services_weekly.csv", id_prefix="tab1-")
    dashboard2 = StaffingAnalysisDashboard(id_prefix="tab2-")
    dashboard3 = StaffMetricsDashboard(services_csv="services_weekly.csv", schedule_csv="staff_schedule.csv",
                                       id_prefix="tab4-")
    dashboard4 = ServicePerformanceDashboard("services_weekly.csv", id_prefix="tab3-")

    # Create the main layout with enhanced title styling
    app.layout = html.Div([
        # Enhanced title section with gradient background and styling
        html.Div([
            html.H1("🏥 Hospital Dashboard",
                    style={
                        "textAlign": "center",
                        "margin": "0",
                        "padding": "30px 20px",
                        "color": "white",
                        "fontSize": "48px",
                        "fontWeight": "bold",
                        "textShadow": "2px 2px 4px rgba(0,0,0,0.3)",
                        "letterSpacing": "2px"
                    })
        ], style={
            "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "borderRadius": "0 0 20px 20px",
            "marginBottom": "30px",
            "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
        }),

        dcc.Tabs(id="tabs", value='tab-1',
                 style={"height": "44px"},
                 parent_style={"marginBottom": "20px"},
                 children=[
            dcc.Tab(label='Task 1', value='tab-1',
                    style=tab_style, selected_style=tab_selected_style,
                    children=[dashboard1.get_layout()]),
            dcc.Tab(label='Operational KPIs Tracking', value='tab-2',
                    style=tab_style, selected_style=tab_selected_style,
                    children=[dashboard2.get_layout()]),
            dcc.Tab(label='Staff Metrics Dashboard', value='tab-3',
                    style=tab_style, selected_style=tab_selected_style,
                    children=[dashboard3.get_layout()]),
            dcc.Tab(label='Service Performance Comparison', value='tab-4',
                    style=tab_style, selected_style=tab_selected_style,
                    children=[dashboard4.get_layout()]),
        ])
    ])

    # Register callbacks for all 4 dashboards
    dashboard1.register_callbacks(app)
    dashboard2.register_callbacks(app)
    dashboard3.register_callbacks(app)
    dashboard4.register_callbacks(app)

    # Run the app
    app.run(debug=True)