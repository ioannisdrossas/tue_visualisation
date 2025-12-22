if __name__ == "__main__":
    from dash import Dash, dcc, html
    from main_final_class import ServicePerformanceDashboard

    # Create the Dash app
    app = Dash(__name__)

    # Create 4 dashboard instances with unique ID prefixes
    dashboard1 = ServicePerformanceDashboard("services_weekly.csv", id_prefix="tab1-")
    dashboard2 = ServicePerformanceDashboard("services_weekly.csv", id_prefix="tab2-")
    dashboard3 = ServicePerformanceDashboard("services_weekly.csv", id_prefix="tab3-")
    dashboard4 = ServicePerformanceDashboard("services_weekly.csv", id_prefix="tab4-")

    # Create the main layout with tabs
    app.layout = html.Div([
        html.H1("Multi-Tab Service Performance Dashboard",
                style={"textAlign": "center", "marginTop": "20px", "color": "#2c3e50"}),

        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='Dashboard 1', value='tab-1', children=[
                dashboard1.get_layout()
            ]),
            dcc.Tab(label='Dashboard 2', value='tab-2', children=[
                dashboard2.get_layout()
            ]),
            dcc.Tab(label='Dashboard 3', value='tab-3', children=[
                dashboard3.get_layout()
            ]),
            dcc.Tab(label='Dashboard 4', value='tab-4', children=[
                dashboard4.get_layout()
            ]),
        ])
    ])

    # Register callbacks for all 4 dashboards
    dashboard1.register_callbacks(app)
    dashboard2.register_callbacks(app)
    dashboard3.register_callbacks(app)
    dashboard4.register_callbacks(app)

    # Run the app
    app.run_server(debug=True)