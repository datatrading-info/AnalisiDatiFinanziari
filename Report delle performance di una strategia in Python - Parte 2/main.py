
import os
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
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
        monthly_table = self.create_monthly_table(self.equity_curve.pct_change().dropna(), 1)
        equity_curve_ffn_stats = self.get_ffn_stats(self.equity_curve)
        kpi_table = self.create_kpi_table(equity_curve_ffn_stats)
        html_out = template.render(perf_chart=perf_chart, drawdown_chart=drawdown_chart, monthly_table=monthly_table,
                                   kpi_table=kpi_table)
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
            name='strategy',
            yaxis='y2',
            line=dict(color=('rgb(22, 96, 167)')))

        trace_benchmark = go.Scatter(
            x=self.benchmark_curve.index.tolist(),
            y=self.rebase_series(self.benchmark_curve).values.tolist(),
            name='benchmark',
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

        return (perf_chart)

    def plot_drawdown_chart(self):

        trace_equity_drawdown = go.Scatter(
            x=self.equity_curve.to_drawdown_series().index.tolist(),
            y=self.equity_curve.to_drawdown_series().values.tolist(),
            name='strategy drawdown',
            yaxis='y2',
            line=dict(color=('rgb(22, 96, 167)')))

        trace_benchmark_drawdown = go.Scatter(
            x=self.benchmark_curve.to_drawdown_series().index.tolist(),
            y=self.benchmark_curve.to_drawdown_series().values.tolist(),
            name='benchmark drawdown',
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

        return (drawdown_chart)

    def create_monthly_table(self, return_series, num_of_compenents):
        return_series.rename('weighted rets', inplace=True)
        return_series = (return_series / float(num_of_compenents))
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
        kpi_table2 = kpi_table.loc[['total_return', 'cagr', 'daily_sharpe',
                                    'daily_vol', 'max_drawdown', 'avg_drawdown']]  # .astype(float)
        kpi_table2['Value'] = pd.Series(["{0:.2f}%".format(val * 100) for val in kpi_table2['Value']],
                                        index=kpi_table2.index)
        kpi_table2.loc['avg_drawdown_days'] = kpi_table.loc['avg_drawdown_days']
        return kpi_table2.to_html(classes="table table-hover table-bordered table-striped", header=False)


if __name__ == "__main__":
    report = PerformanceReport('data.csv')
    report.generate_html_report()
