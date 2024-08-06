import dash
from dash import Dash, html, dash_table
import polars as pl
import altair as alt
import dash_vega_components as dvc
import dash_mantine_components as dmc
import plotly.express as px

dash._dash_renderer._set_react_version("18.2.0")

app = Dash(__name__)

df = pl.read_csv(
    "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
)
df = px.data.tips()
chart = (
    alt.Chart(df)
    .mark_circle(size=60)
    .encode(
        x="tip",
        y="total_bill",
        color=alt.Color("day").scale(domain=["Thur", "Fri", "Sat", "Sun"]),
        tooltip=["day", "tip", "total_bill"],
    )
    .interactive()
)

app.layout = [
    html.H1("Vega-Altair Chart in a Dash App"),
    dvc.Vega(
        id="altair-chart",
        opt={"renderer": "svg", "actions": False},
        spec=chart.to_dict(),
    ),
    html.Div(children="Hello World"),
    dash_table.DataTable(data=df.to_dict("records"), page_size=10),
    dmc.MantineProvider(
        dmc.Popover(
            [
                dmc.PopoverTarget(dmc.Button("Toggle Popover")),
                dmc.PopoverDropdown(dmc.Text("This popover is opened on button click")),
            ],
            width=200,
            position="bottom",
            withArrow=True,
            shadow="md",
            zIndex=2000,
        )
    ),
]

app.layout = dmc.MantineProvider(
    dmc.AppShell(
        [
            dmc.AppShellHeader("Header", px=25),
            dmc.AppShellNavbar("Navbar"),
            dmc.AppShellAside("Aside", withBorder=False),
            dmc.AppShellMain(
                children=[
                    dmc.Text("Some placeholder Text"),
                    dmc.Text("Another placeholder Text"),
                ]
            ),
        ],
        header={"height": 70},
        padding="xl",
        zIndex=1400,
        navbar={
            "width": 300,
            "breakpoint": "sm",
            "collapsed": {"mobile": True},
        },
        aside={
            "width": 300,
            "breakpoint": "xl",
            "collapsed": {"desktop": False, "mobile": True},
        },
    )
)


if __name__ == "__main__":
    app.run(debug=True)
