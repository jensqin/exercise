from dash import Dash, html, dash_table
import polars as pl
# from plotly import express as px

app = Dash()

df = pl.read_csv(
    "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
)
# df = px.data.iris()
app.layout = [
    html.Div(children="Hello World"),
    dash_table.DataTable(data=df.to_dicts(), page_size=10),
]

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
    print()
