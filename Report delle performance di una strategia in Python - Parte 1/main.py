
import os
import pandas as pd
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
        html_out = template.render(perf_chart=perf_chart, drawdown_chart=drawdown_chart)
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


if __name__ == "__main__":
    report = PerformanceReport('data.csv')
    report.generate_html_report()
