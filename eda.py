import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, dash_table
import webbrowser as wb
from threading import Timer

df = pd.read_csv("assets/Amazon Sales data.csv")

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])
df = df.drop_duplicates(subset="Order ID")
df["Units Sold"] = df["Units Sold"].fillna(df["Units Sold"].median())
df["Order Priority"] = df["Order Priority"].fillna(df["Order Priority"].mode()[0])

numeric_cols = [
    "Unit Price",
    "Unit Cost",
    "Total Revenue",
    "Total Cost",
    "Total Profit",
]
for col in numeric_cols:
    df[col] = df[col].astype(float)

cdf = df[(np.abs(stats.zscore(df["Total Profit"])) < 3)]

cdf["Order Date"] = pd.to_datetime(cdf["Order Date"])
cdf["Day"] = cdf["Order Date"].dt.day
cdf["Month"] = cdf["Order Date"].dt.month
cdf["Year"] = cdf["Order Date"].dt.year

monthly_revenue = cdf.groupby(["Year", "Month"])["Total Revenue"].sum().reset_index()
yearly_revenue = cdf.groupby("Year")["Total Revenue"].sum().reset_index()

region_country_revenue = (
    cdf.groupby(["Region", "Country"])["Total Revenue"].sum().reset_index()
)


def calculate_sales_metrics(data):
    total_revenue = data["Total Revenue"].sum()
    total_profit = data["Total Profit"].sum()
    total_units_sold = data["Units Sold"].sum()
    average_order_value = data["Total Revenue"].mean()
    profit_margin = (data["Total Profit"].sum() / data["Total Revenue"].sum()) * 100
    return (
        total_revenue,
        total_profit,
        total_units_sold,
        average_order_value,
        profit_margin,
    )


def regional_and_country_performance(data):
    region_performance = (
        data.groupby("Region")["Total Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
    )
    country_performance = (
        data.groupby("Country")["Total Profit"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
    )
    return region_performance, country_performance


def product_and_sales_channel_insights(data):
    item_performance = (
        data.groupby("Item Type")["Units Sold"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
    )
    sales_channel_revenue = data.groupby("Sales Channel")["Total Revenue"].sum()
    return item_performance, sales_channel_revenue


def order_and_shipping_efficiency(data):
    order_priority_revenue = data.groupby("Order Priority")["Total Revenue"].sum()
    average_shipping_time = (data["Ship Date"] - data["Order Date"]).dt.days.mean()
    return order_priority_revenue, average_shipping_time


total_revenue, total_profit, total_units_sold, average_order_value, profit_margin = (
    calculate_sales_metrics(cdf)
)
region_performance, country_performance = regional_and_country_performance(cdf)
item_performance, sales_channel_revenue = product_and_sales_channel_insights(cdf)
order_priority_revenue, average_shipping_time = order_and_shipping_efficiency(cdf)

numeric_columns = cdf.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()

fig_corr = px.imshow(
    correlation_matrix,
    text_auto=True,
    aspect="equal",
    height=800,
    title="Correlation Matrix",
    color_continuous_scale="Viridis",
)

findings = []

threshold = 0.7
for col in correlation_matrix.columns:
    for idx in correlation_matrix.index:
        if col != idx and abs(correlation_matrix.loc[idx, col]) > threshold:
            findings.append(
                f"{idx} and {col} have a significant correlation of {correlation_matrix.loc[idx, col]:.2f}"
            )

monthly_colors = [
    "#FF6347",
    "#FFD700",
    "#ADFF2F",
    "#20B2AA",
    "#1E90FF",
    "#BA55D3",
    "#FF69B4",
    "#FFA07A",
    "#8B0000",
    "#00FF00",
    "#4B0082",
    "#00CED1",
]
yearly_colors = [
    "#FF6347",
    "#FFD700",
    "#ADFF2F",
    "#20B2AA",
    "#1E90FF",
    "#BA55D3",
    "#FF69B4",
    "#FFA07A",
    "#8B0000",
]

monthly_revenue_sum = cdf.groupby("Month")["Total Revenue"].sum().reset_index()
yearly_revenue_sum = cdf.groupby("Year")["Total Revenue"].sum().reset_index()

mon_rev = monthly_revenue.round(2).astype(str)
yer_rev = yearly_revenue.round(2).astype(str)

msd_fig = go.Figure(
    data=[
        go.Pie(
            labels=[
                f"Month: {month}, Revenue: ${revenue:,.2f}, Percentage: {percent:.1f}%"
                for month, revenue, percent in zip(
                    monthly_revenue_sum["Month"],
                    monthly_revenue_sum["Total Revenue"],
                    100
                    * monthly_revenue_sum["Total Revenue"]
                    / monthly_revenue_sum["Total Revenue"].sum(),
                )
            ],
            values=monthly_revenue_sum["Total Revenue"],
            hoverinfo="label",
            textinfo="label",
            marker=dict(colors=monthly_colors),
        )
    ]
)
msd_fig.update_layout(title="Monthly Sales Distribution", height=500, showlegend=False)

ysd_fig = go.Figure(
    data=[
        go.Pie(
            labels=[
                f"Year: {year}, Revenue: ${revenue}, Percentage: {percent:.1f}%"
                for year, revenue, percent in zip(
                    yearly_revenue["Year"],
                    yearly_revenue["Total Revenue"],
                    100
                    * yearly_revenue["Total Revenue"].astype(float)
                    / yearly_revenue["Total Revenue"].astype(float).sum(),
                )
            ],
            values=yearly_revenue["Total Revenue"].astype(float),
            hoverinfo="label",
            textinfo="label",
            marker=dict(colors=yearly_colors),
        )
    ]
)
ysd_fig.update_layout(title="Yearly Sales Distribution", height=500, showlegend=False)

rc_fig = go.Figure()
rc_fig.add_trace(
    go.Bar(
        x=region_country_revenue["Country"],
        y=region_country_revenue["Total Revenue"],
        name="Sales by Region and Country",
        marker=dict(
            color=region_country_revenue["Total Revenue"], colorscale="Viridis"
        ),
        hoverinfo="y+text",
        text=[
            f"Region: {region}, Country: {country}<br>Total Revenue: ${revenue:,.2f}"
            for region, country, revenue in zip(
                region_country_revenue["Region"],
                region_country_revenue["Country"],
                region_country_revenue["Total Revenue"],
            )
        ],
    )
)
rc_fig.update_layout(
    title="Sales by Region and Country",
    xaxis_title="Country",
    yaxis_title="Total Revenue ($)",
    height=800,
    showlegend=False,
    coloraxis=dict(
        colorscale="Viridis",
        colorbar=dict(title="Total Revenue ($)", tickformat="$,.2f"),
    ),
    bargap=0.2,
)
fig_monthly_trends = go.Figure()
fig_monthly_trends.add_trace(
    go.Scatter(
        x=monthly_revenue["Year"].astype(str)
        + "-"
        + monthly_revenue["Month"].astype(str),
        y=monthly_revenue["Total Revenue"],
        mode="lines+markers",
        line=dict(color="blue"),
        name="Monthly Revenue",
    )
)
fig_monthly_trends.update_layout(
    title="Monthly Sales Trends",
    xaxis_title="Year-Month",
    yaxis_title="Revenue",
    template="plotly",
    height=600,
)

fig_yearly_trends = go.Figure()
fig_yearly_trends.add_trace(
    go.Scatter(
        x=yearly_revenue["Year"],
        y=yearly_revenue["Total Revenue"],
        mode="lines+markers",
        line=dict(color="green"),
        name="Yearly Revenue",
    )
)
fig_yearly_trends.update_layout(
    title="Yearly Sales Trends",
    xaxis_title="Year",
    yaxis_title="Revenue",
    template="plotly",
    height=600,
)

monthly_revenue_diff = monthly_revenue["Total Revenue"].diff()
fig_monthly_revenue_change = go.Figure()
fig_monthly_revenue_change.add_trace(
    go.Scatter(
        x=monthly_revenue["Year"].astype(str)
        + "-"
        + monthly_revenue["Month"].astype(str),
        y=monthly_revenue_diff,
        mode="lines+markers",
        line=dict(color="orange"),
        name="Monthly Revenue Change",
    )
)
fig_monthly_revenue_change.add_hline(y=0, line_dash="dash", line_color="black")
fig_monthly_revenue_change.update_layout(
    title="Monthly Revenue Change",
    xaxis_title="Year-Month",
    yaxis_title="Revenue Change",
    template="plotly",
    height=600,
)

yearly_revenue_change = yearly_revenue["Total Revenue"].diff()

fig_yearly_revenue_change = go.Figure()
fig_yearly_revenue_change.add_trace(
    go.Scatter(
        x=yearly_revenue["Year"],
        y=yearly_revenue_change,
        mode="lines+markers",
        line=dict(color="purple"),
        name="Yearly Revenue Change",
    )
)
fig_yearly_revenue_change.add_hline(y=0, line_dash="dash", line_color="black")
fig_yearly_revenue_change.update_layout(
    title="Yearly Revenue Change",
    xaxis_title="Year",
    yaxis_title="Revenue Change",
    template="plotly",
    height=600,
)

fig_monthly_high_low = go.Figure()
fig_monthly_high_low.add_trace(
    go.Bar(
        x=monthly_revenue["Year"].astype(str)
        + "-"
        + monthly_revenue["Month"].astype(str),
        y=monthly_revenue["Total Revenue"],
        marker=dict(color=monthly_revenue["Total Revenue"], colorscale="Viridis"),
        name="Monthly Revenue",
    )
)
max_monthly_sales = monthly_revenue["Total Revenue"].max()
min_monthly_sales = monthly_revenue["Total Revenue"].min()
fig_monthly_high_low.add_hline(
    y=max_monthly_sales,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Highest: {max_monthly_sales:.2f}",
)
fig_monthly_high_low.add_hline(
    y=min_monthly_sales,
    line_dash="dash",
    line_color="blue",
    annotation_text=f"Lowest: {min_monthly_sales:.2f}",
)
fig_monthly_high_low.update_layout(
    title="Monthly Sales with Highest and Lowest Points",
    xaxis_title="Year-Month",
    yaxis_title="Revenue",
    template="plotly",
    height=600,
)

fig_yearly_high_low = go.Figure()
fig_yearly_high_low.add_trace(
    go.Bar(
        x=yearly_revenue["Year"],
        y=yearly_revenue["Total Revenue"],
        marker=dict(color=yearly_revenue["Total Revenue"], colorscale="Viridis"),
        name="Yearly Revenue",
    )
)
max_yearly_sales = yearly_revenue["Total Revenue"].max()
min_yearly_sales = yearly_revenue["Total Revenue"].min()
fig_yearly_high_low.add_hline(
    y=max_yearly_sales,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Highest: {max_yearly_sales:.2f}",
)
fig_yearly_high_low.add_hline(
    y=min_yearly_sales,
    line_dash="dash",
    line_color="blue",
    annotation_text=f"Lowest: {min_yearly_sales:.2f}",
)
fig_yearly_high_low.update_layout(
    title="Yearly Sales with Highest and Lowest Points",
    xaxis_title="Year",
    yaxis_title="Revenue",
    template="plotly",
    height=600,
)

app = Dash(__name__, title="Amazon Sales Analysis")

app.layout = html.Div(
    [
        html.Div([html.H1("Amazon Sales Analysis")], id="header-container"),
        #! ----             ----                ----                ----                ----
        # * 1. Overview
        html.Div(
            [
                html.H2("1. Overview"),
                html.P("Quick snippets of the analysis.", className="hed sub-hed"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("1. Sales Metrics", className="hed"),
                                html.P(
                                    "Quick snippets of the analysis.",
                                    className="hed sub-hed",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P(
                                                    "Total Units Sold", className="hed"
                                                ),
                                                html.H4(
                                                    f"{total_units_sold:,}",
                                                    className="val",
                                                ),
                                            ],
                                            className="insights",
                                        ),
                                        html.Div(
                                            [
                                                html.P(
                                                    "Average Order Value",
                                                    className="hed",
                                                ),
                                                html.H4(
                                                    f"${average_order_value:,.2f}",
                                                    className="val",
                                                ),
                                            ],
                                            className="insights",
                                        ),
                                        html.Div(
                                            [
                                                html.P(
                                                    "Total Revenue", className="hed"
                                                ),
                                                html.H4(
                                                    f"${total_revenue:,.2f}",
                                                    className="val",
                                                ),
                                            ],
                                            className="insights",
                                        ),
                                        html.Div(
                                            [
                                                html.P("Total Profit", className="hed"),
                                                html.H4(
                                                    f"${total_profit:,.2f}",
                                                    className="val",
                                                ),
                                            ],
                                            className="insights",
                                        ),
                                        html.Div(
                                            [
                                                html.P(
                                                    "Profit Margin", className="hed"
                                                ),
                                                html.H4(
                                                    f"{profit_margin:.2f}%",
                                                    className="val",
                                                ),
                                            ],
                                            className="insights",
                                        ),
                                    ],
                                    className="i-c",
                                ),
                            ],
                            className="ins-con",
                        ),
                        html.Div(
                            [
                                html.H3(
                                    "2. Regional and Country Performance",
                                    className="hed",
                                ),
                                html.P(
                                    "Top 3 Regions by Revenue", className="hed sub-hed"
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P(region, className="hed"),
                                                html.H4(
                                                    f"${revenue:,.2f}", className="val"
                                                ),
                                            ],
                                            className="insights",
                                        )
                                        for region, revenue in region_performance.items()
                                    ],
                                    className="i-c",
                                ),
                                html.P(
                                    "Top 3 Countries by Profit", className="hed sub-hed"
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P(country, className="hed"),
                                                html.H4(
                                                    f"${profit:,.2f}", className="val"
                                                ),
                                            ],
                                            className="insights",
                                        )
                                        for country, profit in country_performance.items()
                                    ],
                                    className="i-c",
                                ),
                            ],
                            className="ins-con",
                        ),
                        html.Div(
                            [
                                html.H3(
                                    "3. Products and Sales Channel Insights",
                                    className="hed",
                                ),
                                html.P(
                                    "Top 3 Best Selling Item Types",
                                    className="hed sub-hed",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P(item, className="hed"),
                                                html.H3(
                                                    f"{units_sold:,}", className="val"
                                                ),
                                            ],
                                            className="insights",
                                        )
                                        for item, units_sold in item_performance.items()
                                    ],
                                    className="i-c",
                                ),
                                html.P(
                                    "Revenue by Sales Channel", className="hed sub-hed"
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P(channel, className="hed"),
                                                html.H3(
                                                    f"${revenue:,.2f}", className="val"
                                                ),
                                            ],
                                            className="insights",
                                        )
                                        for channel, revenue in sales_channel_revenue.items()
                                    ],
                                    className="i-c",
                                ),
                            ],
                            className="ins-con",
                        ),
                        html.Div(
                            [
                                html.H3(
                                    "4. Order and Shipping Efficiency", className="hed"
                                ),
                                html.P(
                                    "Order Priority Impact on Revenue by Critical, High, Medium & Low",
                                    className="hed sub-hed",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P(priority, className="hed"),
                                                html.H4(
                                                    f"${revenue:,.2f}", className="val"
                                                ),
                                            ],
                                            className="insights",
                                        )
                                        for priority, revenue in order_priority_revenue.items()
                                    ],
                                    className="i-c",
                                ),
                                html.P(
                                    "Average Shipping Time", className="hed sub-hed"
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P("Days", className="hed"),
                                                html.H4(
                                                    f"{average_shipping_time:.2f}",
                                                    className="val",
                                                ),
                                            ],
                                            className="insights",
                                        )
                                    ],
                                    className="i-c",
                                ),
                            ],
                            className="ins-con",
                        ),
                        html.Div(
                            [
                                html.H3("5. Quick Visuals", className="hed"),
                                html.P(
                                    "Here are the visualizations from the analysis",
                                    className="hed sub-hed",
                                ),
                                dcc.Graph(figure=fig_monthly_trends),
                                dcc.Graph(figure=fig_monthly_revenue_change),
                                dcc.Graph(figure=fig_yearly_trends),
                                dcc.Graph(figure=fig_yearly_revenue_change),
                                dcc.Graph(figure=fig_monthly_high_low),
                                dcc.Graph(figure=fig_yearly_high_low),
                            ],
                            className="ins-con",
                        ),
                    ],
                    className="insights-container",
                ),
            ],
            id="overview-container",
        ),
        #! ----             ----                ----                ----                ----
        # * 2. Monthly & Yearly Sales Distribution
        html.Div(
            [
                html.H2("2. Monthly & Yearly Sales Distribution"),
                html.P(
                    "Hover over the segments to see detailed revenue and percentage contribution.",
                    className="hed sub-hed",
                ),
                dcc.Graph(figure=msd_fig),
                dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in mon_rev.columns],
                    data=mon_rev.to_dict("records"),
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "rgb(230, 230, 230)",
                        "fontWeight": "bold",
                    },
                    style_cell={"textAlign": "center", "padding": "5px"},
                    page_size=100,
                ),
                dcc.Graph(figure=ysd_fig),
                dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in yer_rev.columns],
                    data=yer_rev.to_dict("records"),
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "rgb(230, 230, 230)",
                        "fontWeight": "bold",
                    },
                    style_cell={"textAlign": "center", "padding": "5px"},
                    page_size=100,
                ),
            ],
            id="sales-distribution-container",
        ),
        #! ----             ----                ----                ----                ----
        # * 3. Sales Trends
        html.Div(
            [
                html.H2("3. Sales Trends"),
                html.P(
                    "Choose the option from the dropdown list below to see detailed information of Monthly by year, Monthly, Yearly and Regionally sales trend data.",
                    className="hed sub-hed",
                ),
                dcc.Dropdown(
                    id="sales-trend-dropdown",
                    options=[
                        {"label": "Monthly Sales by Year", "value": "monthly-by-year"},
                        {"label": "Monthly Sales Trends", "value": "monthly"},
                        {"label": "Yearly Sales Trends", "value": "yearly"},
                        {
                            "label": "Sales by Region and Country",
                            "value": "region_country",
                        },
                    ],
                    value="monthly-by-year",
                    className="dash-dropdown sub-hed",
                ),
                html.Div(id="sales-trends"),
            ],
            id="sales-trend-container",
        ),
        #! ----             ----                ----                ----                ----
        # * 4. Correlation Matrix
        html.Div(
            [
                html.H2("4. Correlation Matrix"),
                html.P(
                    "",
                    className="hed sub-hed",
                ),
                dcc.Graph(figure=fig_corr),
                html.H3(
                    "Findings from Correlation Matrix",
                    className="hed",
                ),
                html.Ul([html.Li(finding) for finding in findings]),
            ],
            className="correlations-container",
        ),
        #! ----             ----                ----                ----                ----
        # * 5. Conclusion & Findings
        html.Div(
            [
                html.H2("5. Conclusion & Findings"),
                html.P(
                    "Based on the analysis of the sales data, we can draw the following conclusions:",
                    className="hed sub-hed",
                ),
                html.Ul(
                    [
                        html.Li(
                            f"The total revenue generated over the analyzed period is ${total_revenue:,.2f}."
                        ),
                        html.Li(
                            f"The total profit earned is ${total_profit:,.2f}%, indicating a profit margin of {profit_margin:.2f}%."
                        ),
                        html.Li(
                            f"A total of {total_units_sold:,} units were sold, demonstrating strong sales volume."
                        ),
                        html.Li(
                            f"The highest monthly sales occurred in Month {monthly_revenue.loc[monthly_revenue['Total Revenue'].astype(float).idxmax()]['Month']} of Year {monthly_revenue.loc[monthly_revenue['Total Revenue'].astype(float).idxmax()]['Year']}."
                        ),
                        html.Li(
                            f"The lowest monthly sales were recorded in Month {monthly_revenue.loc[monthly_revenue['Total Revenue'].astype(float).idxmin()]['Month']} of Year {monthly_revenue.loc[monthly_revenue['Total Revenue'].astype(float).idxmin()]['Year']}."
                        ),
                        html.Li(
                            f"The overall trend shows a significant increase in sales year-over-year, with Year {yearly_revenue.loc[yearly_revenue['Total Revenue'].astype(float).idxmax()]['Year']} having the highest revenue."
                        ),
                        html.Li(
                            f"The overall trend shows a significant decrease in sales year-over-year, with Year {yearly_revenue.loc[yearly_revenue['Total Revenue'].astype(float).idxmin()]['Year']} having the lowest revenue."
                        ),
                    ]
                ),
            ],
            id="conclusion-container",
        ),
    ],
    id="main-container",
)


@app.callback(
    Output("sales-trends", "children"),
    Input("sales-trend-dropdown", "value"),
)
def update_sales_trend(selected_trend):
    graphs = []
    if selected_trend == "monthly-by-year":
        for year in cdf["Year"].unique():
            filtered_data = monthly_revenue[monthly_revenue["Year"] == year]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=filtered_data["Month"],
                    y=filtered_data["Total Revenue"].astype(float),
                    name=f"Monthly Sales in {year}",
                    marker=dict(
                        color=filtered_data["Total Revenue"].astype(float),
                        colorscale="Viridis",
                    ),
                    hoverinfo="y+text",
                    text=[
                        f"Month: {month}<br>Total Revenue: ${revenue:.2f}<br>Units Sold: {units_sold}<br>Avg. Unit Price: ${unit_price:.2f}<br>Total Cost: ${total_cost:.2f}<br>Total Profit: ${total_profit:.2f}"
                        for month, revenue, units_sold, unit_price, total_cost, total_profit in zip(
                            filtered_data["Month"],
                            filtered_data["Total Revenue"].astype(float),
                            cdf.groupby("Month")["Units Sold"].sum(),
                            cdf.groupby("Month")["Unit Price"].mean(),
                            cdf.groupby("Month")["Total Cost"].sum(),
                            cdf.groupby("Month")["Total Profit"].sum(),
                        )
                    ],
                )
            )
            fig.update_layout(
                title=f"Monthly Sales Trends for {year}",
                xaxis_title="Month",
                yaxis_title="Total Revenue ($)",
                height=600,
                showlegend=False,
                coloraxis=dict(
                    colorscale="Viridis",
                    colorbar=dict(title="Total Revenue ($)", tickformat="$,.2f"),
                ),
                bargap=0.2,
            )
            graphs.append(dcc.Graph(figure=fig))
    elif selected_trend == "monthly":
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=monthly_revenue["Month"],
                y=monthly_revenue["Total Revenue"].astype(float),
                name="Monthly Sales",
                marker=dict(
                    color=monthly_revenue["Total Revenue"].astype(float),
                    colorscale="Viridis",
                ),
                hoverinfo="y+text",
                text=[
                    f"Month: {month}<br>Total Revenue: ${revenue:.2f}"
                    for month, revenue in zip(
                        monthly_revenue["Month"],
                        monthly_revenue["Total Revenue"].astype(float),
                    )
                ],
            )
        )
        fig.update_layout(
            title="Monthly Sales Trends",
            xaxis_title="Month",
            yaxis_title="Total Revenue ($)",
            height=600,
            showlegend=False,
            coloraxis=dict(
                colorscale="Viridis",
                colorbar=dict(title="Total Revenue ($)", tickformat="$,.2f"),
            ),
            bargap=0.2,
        )
        graphs.append(dcc.Graph(figure=fig))
    elif selected_trend == "yearly":
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=yearly_revenue["Year"],
                y=yearly_revenue["Total Revenue"].astype(float),
                name="Yearly Sales",
                marker=dict(
                    color=yearly_revenue["Total Revenue"].astype(float),
                    colorscale="Viridis",
                ),
                hoverinfo="y+text",
                text=[
                    f"Year: {year}<br>Total Revenue: ${revenue:.2f}"
                    for year, revenue in zip(
                        yearly_revenue["Year"],
                        yearly_revenue["Total Revenue"].astype(float),
                    )
                ],
            )
        )
        fig.update_layout(
            title="Yearly Sales Trends",
            xaxis_title="Year",
            yaxis_title="Total Revenue ($)",
            height=600,
            showlegend=False,
            coloraxis=dict(
                colorscale="Viridis",
                colorbar=dict(title="Total Revenue ($)", tickformat="$,.2f"),
            ),
            bargap=0.2,
        )
        graphs.append(dcc.Graph(figure=fig))
    elif selected_trend == "region_country":
        region_country_fig = go.Figure()
        for region in region_country_revenue["Region"].unique():
            region_data = region_country_revenue[
                region_country_revenue["Region"] == region
            ]
            region_country_fig.add_trace(
                go.Bar(
                    x=region_data["Country"],
                    y=region_data["Total Revenue"],
                    name=f"Sales in {region}",
                    marker=dict(
                        color=region_data["Total Revenue"], colorscale="Viridis"
                    ),
                    hoverinfo="y+text",
                    text=[
                        f"Region: {region}<br>Country: {country}<br>Total Revenue: ${revenue:.2f}"
                        for country, revenue in zip(
                            region_data["Country"], region_data["Total Revenue"]
                        )
                    ],
                )
            )
        region_country_fig.update_layout(
            title="Sales by Region and Country",
            xaxis_title="Country",
            yaxis_title="Total Revenue ($)",
            height=600,
            showlegend=True,
            coloraxis=dict(
                colorscale="Viridis",
                colorbar=dict(title="Total Revenue ($)", tickformat="$,.2f"),
            ),
            bargap=0.2,
        )
        graphs.append(dcc.Graph(figure=region_country_fig))
    return graphs


def open_in_browser(app):
    Timer(1, lambda: wb.open("http://127.0.0.1:8050/")).start()
    app.run_server(debug=False, port=8050, host="0.0.0.0")


if __name__ == "__main__":
    open_in_browser(app)
