import os
import math
import pandas as pd
import numpy as np
import random
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
import ffn
from jinja2 import Environment, FileSystemLoader


class PerformanceReport:
    """ Report con le statistiche delle performance stats per una data strategia
    """

    def __init__(self, infilename):
        self.infilename = infilename
        self.get_data()

    def get_data(self):
        basedir = os.path.abspath(os.path.dirname('__file__'))
        data_folder = os.path.join(basedir, 'data')
        data = pd.read_csv(os.path.join(data_folder, self.infilename), index_col='date',
                           parse_dates=True, dayfirst=True)
        self.equity_curve = data['equity_curve']

        if len(data.columns) > 1:
            self.benchmark_curve = data['benchmark']

    def generate_html(self):
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template("templates/template.html")
        perf_chart = self.plot_performance_chart()
        drawdown_chart = self.plot_drawdown_chart()
        monthly_table = self.create_monthly_table(self.equity_curve.pct_change().dropna())
        equity_curve_ffn_stats = self.get_ffn_stats(self.equity_curve)
        benchmark_curve_ffn_stats = self.get_ffn_stats(self.benchmark_curve)
        kpi_table = self.create_kpi_table(equity_curve_ffn_stats)
        kpi_table_full = self.create_kpi_table_full([equity_curve_ffn_stats, benchmark_curve_ffn_stats])
        kpi_table_1, kpi_table_2, kpi_table_3, kpi_table_4, kpi_table_5 = self.split_kpi_table(kpi_table_full)
        daily_ret_hist = self.plot_daily_histogram()
        daily_ret_box = self.plot_daily_box()
        simulations = 250
        periods = 252
        monte_carlo_results, monte_carlo_hist = self.run_monte_carlo_parametric(self.equity_curve.pct_change().dropna(),
                                                                                periods, simulations)
        mc_chart = self.plot_mc_chart(monte_carlo_results)
        mc_dist_chart = self.plot_mc_dist_chart(monte_carlo_hist)
        mc_hist_chart = self.plot_mc_hist_chart(monte_carlo_hist)

        html_out = template.render(perf_chart=perf_chart, drawdown_chart=drawdown_chart, monthly_table=monthly_table,
                                   kpi_table=kpi_table, kpi_table_1=kpi_table_1, kpi_table_2=kpi_table_2,
                                   kpi_table_3=kpi_table_3, kpi_table_4=kpi_table_4, kpi_table_5=kpi_table_5,
                                   daily_ret_hist=daily_ret_hist, daily_ret_box=daily_ret_box, mc_chart=mc_chart,
                                   mc_dist_chart=mc_dist_chart, mc_hist_chart=mc_hist_chart)
        return html_out

    def generate_html_report(self):
        """ Restitusice un report HTML con le analisi
        """
        html = self.generate_html()
        outputdir = "output"
        outfile = os.path.join(outputdir, 'report.html')
        file = open(outfile, "w")
        file.write(html)
        file.close()

    def rebase_series(self, series):
        return (series / series.iloc[0]) * 100

    def plot_performance_chart(self):

        trace_equity = go.Scatter(
            x=self.equity_curve.index.tolist(),
            y=self.rebase_series(self.equity_curve).values.tolist(),
            name='Strategy',
            yaxis='y2',
            line=dict(color=('rgb(22, 96, 167)')))

        trace_benchmark = go.Scatter(
            x=self.benchmark_curve.index.tolist(),
            y=self.rebase_series(self.benchmark_curve).values.tolist(),
            name='Benchmark',
            yaxis='y2',
            line=dict(color=('rgb(22, 96, 0)')))

        layout = go.Layout(
            autosize=True,
            legend=dict(orientation="h"),
            title='Performance Chart',
            yaxis=dict(
                title='Performance'))

        perf_chart = plotly.offline.plot({"data": [trace_equity, trace_benchmark],
                                          "layout": layout}, include_plotlyjs=False,
                                         output_type='div')

        return perf_chart

    def plot_drawdown_chart(self):

        trace_equity_drawdown = go.Scatter(
            x=self.equity_curve.to_drawdown_series().index.tolist(),
            y=self.equity_curve.to_drawdown_series().values.tolist(),
            name='Strategy',
            yaxis='y2',
            line=dict(color=('rgb(22, 96, 167)')))

        trace_benchmark_drawdown = go.Scatter(
            x=self.benchmark_curve.to_drawdown_series().index.tolist(),
            y=self.benchmark_curve.to_drawdown_series().values.tolist(),
            name='Benchmark',
            yaxis='y2',
            line=dict(color=('rgb(22, 96, 0)')))

        layout = go.Layout(
            autosize=True,
            legend=dict(orientation="h"),
            title='Drawdown Chart',
            yaxis=dict(
                title='Drawdown'))

        drawdown_chart = plotly.offline.plot({"data": [trace_equity_drawdown, trace_benchmark_drawdown],
                                              "layout": layout}, include_plotlyjs=False,
                                             output_type='div')

        return drawdown_chart

    def create_monthly_table(self, return_series):
        return_series.rename('weighted rets', inplace=True)
        returns_df_m = pd.DataFrame((return_series + 1).resample('M').prod() - 1)
        returns_df_m['Month'] = returns_df_m.index.month
        monthly_table = returns_df_m[['weighted rets', 'Month']].pivot_table(returns_df_m[['weighted rets', 'Month']],
                                                                             index=returns_df_m.index, columns='Month',
                                                                             aggfunc=np.sum).resample('A')
        monthly_table = monthly_table.aggregate('sum')
        monthly_table.columns = monthly_table.columns.droplevel()

        monthly_table.index = monthly_table.index.year
        monthly_table['YTD'] = ((monthly_table + 1).prod(axis=1) - 1)
        monthly_table = monthly_table * 100
        monthly_table.replace(0.0, "", inplace=True)

        monthly_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
                                 'YTD']
        return monthly_table.round(2).fillna("").to_html(classes="table table-hover table-bordered table-striped")

    def get_ffn_stats(self, equity_series):
        equity_stats = equity_series.calc_stats()
        d = dict(equity_stats.stats)
        return d

    def create_kpi_table(self, ffn_dict):
        kpi_table = pd.DataFrame.from_dict(ffn_dict, orient='index')
        kpi_table.index.name = 'KPI'
        kpi_table.columns = ['Value']
        kpi_table2 = kpi_table.loc[['total_return', 'cagr',
                                    'daily_vol', 'max_drawdown', 'avg_drawdown']]  # .astype(float)
        kpi_table2['Value'] = pd.Series(["{0:.2f}%".format(val * 100) for val in kpi_table2['Value']],
                                        index=kpi_table2.index)
        kpi_table2.loc['avg_drawdown_days'] = kpi_table.loc['avg_drawdown_days']
        kpi_table2.loc['daily_sharpe'] = np.round(kpi_table.loc['daily_sharpe'].values[0], 2)
        return kpi_table2.to_html(classes="table table-hover table-bordered table-striped", header=False)

    def create_kpi_table_full(self, ffn_dict_list):
        df_list = [pd.DataFrame.from_dict(x, orient='index') for x in ffn_dict_list]
        kpi_table_full = pd.concat(df_list, axis=1)
        return kpi_table_full

    def split_kpi_table(self, kpi_table_full):
        kpi_table_1 = kpi_table_full.iloc[3:16].to_html(classes="table table-hover table-bordered table-striped",
                                                        header=False)
        kpi_table_2 = kpi_table_full.iloc[16:24].to_html(classes="table table-hover table-bordered table-striped",
                                                         header=False)
        kpi_table_3 = kpi_table_full.iloc[24:32].to_html(classes="table table-hover table-bordered table-striped",
                                                         header=False)
        kpi_table_4 = kpi_table_full.iloc[32:40].to_html(classes="table table-hover table-bordered table-striped",
                                                         header=False)
        kpi_table_5 = kpi_table_full.iloc[40:].to_html(classes="table table-hover table-bordered table-striped",
                                                       header=False)
        return kpi_table_1, kpi_table_2, kpi_table_3, kpi_table_4, kpi_table_5

    def run_monte_carlo_parametric(self, returns, trading_days, simulations):
        df_list = []
        result = []
        S = 100
        T = trading_days
        mu = returns.mean()
        vol = returns.std()

        for i in range(simulations):

            daily_returns = np.random.normal(mu / T, vol / math.sqrt(T), T) + 1

            price_list = [S]

            for x in daily_returns:
                price_list.append(price_list[-1] * x)

            df_list.append(pd.DataFrame(price_list))
            result.append(price_list[-1])
        df_master = pd.concat(df_list, axis=1)
        df_master.columns = range(len(df_master.columns))

        return df_master, result

    def plot_daily_histogram(self):
        trace0 = go.Histogram(x=self.equity_curve.pct_change().values,
                              name="Strategy",
                              opacity=0.75,
                              marker=dict(
                                  color=('rgb(22, 96, 167)')),
                              xbins=dict(
                                  size=0.0025
                              ))
        trace1 = go.Histogram(x=self.benchmark_curve.pct_change().values,
                              name="Benchmark",
                              opacity=0.75,
                              marker=dict(
                                  color=('rgb(22, 96, 0)')),
                              xbins=dict(
                                  size=0.0025
                              ))
        data = [trace0, trace1]
        layout = go.Layout(
            title='Histogram of Strategy and Benchmark Daily Returns',
            autosize=True,
            height=600,
            hovermode='closest',
            barmode='overlay'
        )
        daily_ret_hist = plotly.offline.plot({"data": data, "layout": layout},
                                             include_plotlyjs=False,
                                             output_type='div')

        return daily_ret_hist

    def plot_daily_box(self):
        trace0 = go.Box(y=self.equity_curve.pct_change().values,
                        name="Strategy",
                        marker=dict(
                            color=('rgb(22, 96, 167)')))
        trace1 = go.Box(y=self.benchmark_curve.pct_change().values,
                        name="Benchmark",
                        marker=dict(
                            color=('rgb(22, 96, 0)')))
        data = [trace0, trace1]

        layout = go.Layout(
            title='Boxplot of Strategy and Benchmark Daily Returns',
            autosize=True,
            height=600,
            yaxis=dict(
                zeroline=False
            )
        )

        box_plot = plotly.offline.plot({"data": data, "layout": layout}, include_plotlyjs=False,
                                       output_type='div')

        return box_plot

    def get_random_rgb(self):
        col = 'rgb' + str((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        return col

    def plot_mc_chart(self, monte_carlo_results):
        mc_trace_list = []
        for col in monte_carlo_results.columns:
            rgb = self.get_random_rgb()
            trace = go.Scatter(
                x=monte_carlo_results.index.tolist(),
                y=monte_carlo_results[col].values.tolist(),
                name='mc data',
                yaxis='y2',
                line=dict(color=(rgb)))
            mc_trace_list.append(trace)

        layout_mc = go.Layout(
            title='Monte Carlo Parametric',
            yaxis=dict(title='Equity'),
            autosize=True,
            height=600,
            showlegend=False,
            hovermode='closest'
        )
        mc_chart = plotly.offline.plot({"data": mc_trace_list,
                                        "layout": layout_mc}, include_plotlyjs=False,
                                       output_type='div')
        return mc_chart

    def plot_mc_dist_chart(self, monte_carlo_hist):
        fig = ff.create_distplot([monte_carlo_hist], group_labels=['Strategy Monte Carlo Returns'], show_rug=False)
        fig['layout'].update(autosize=True, height=600,
                             title="Parametric Monte Carlo Simulations <br> (Distribution Plot - Ending Equity)",
                             showlegend=False)
        dist_plot_MC_parametric = plotly.offline.plot(fig, include_plotlyjs=False,
                                                      output_type='div')
        return dist_plot_MC_parametric

    def plot_mc_hist_chart(self, monte_carlo_hist):
        layout_mc = go.Layout(
            title="Parametric Monte Carlo Simulations <br> (Histogram - Ending Equity)",
            autosize=True,
            height=600,
            showlegend=False,
            hovermode='closest'
        )

        data_mc_hist = [go.Histogram(x=monte_carlo_hist,
                                     histnorm='probability')]
        mc_hist = plotly.offline.plot({"data": data_mc_hist, "layout": layout_mc}, include_plotlyjs=False,
                                      output_type='div')
        return mc_hist


if __name__ == "__main__":
    report = PerformanceReport('data.csv')
    report.generate_html_report()