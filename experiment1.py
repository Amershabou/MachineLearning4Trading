import ManualStrategy as ms
import pandas as dt
import StrategyLearner as sl
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals
from util import get_data
import pandas as pd


class experimnent1(object):
    def __init__(self):
        pass

    def author(self):
        return "ashabou3"

    def runExperiment1(self):
        manualStrategy = ms.ManualStrategy()
        sample = "In"

        trades_inSample_manual, trades_inSample_bench = manualStrategy.testPolicy(symbol="JPM",
                                                                                  sd=dt.datetime(2008, 1, 1),
                                                                                  ed=dt.datetime(2009, 12, 31),
                                                                                  sv=100000, sample=sample)
        sample = "Out of"

        trades_outOfSample_manual, trades_outOfSample_bench = manualStrategy.testPolicy(symbol="JPM",
                                                                                        sd=dt.datetime(2010, 1, 1),
                                                                                        ed=dt.datetime(2011, 12, 31),
                                                                                        sv=100000, sample=sample)

        learner = sl.StrategyLearner(verbose=False, impact=0.0, commission=0.0)  # constructor
        learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                             sv=100000)  # training phase

        df_trades_inSample = learner.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                                sv=100000)  # testing phase
        df_trades_outOfSample = learner.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1),
                                                   ed=dt.datetime(2011, 12, 31),
                                                   sv=100000)  # testing phase

        dates_out_of_sample = pd.date_range(dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31))
        prices_out_of_sample = get_data(["JPM"], dates_out_of_sample).drop(columns=['SPY'], axis=1)
        classificationStrategy_portval_inSample = compute_portvals(df_trades_inSample, learner.prices, "JPM",
                                                                   learner.impact, learner.commission, start_val=100000)

        classificationStrategy_portval_inSample = classificationStrategy_portval_inSample / \
                                                  classificationStrategy_portval_inSample[0]

        classificationStrategy_portval_outOfSample = compute_portvals(df_trades_outOfSample, prices_out_of_sample,
                                                                      "JPM", learner.impact, learner.commission,
                                                                      start_val=100000)

        classificationStrategy_portval_outOfSample = classificationStrategy_portval_outOfSample / \
                                                     classificationStrategy_portval_outOfSample[0]

        ax = trades_inSample_manual.plot(fontsize=12, color="red", label="Manual Strategy")
        trades_inSample_bench.plot(ax=ax, color="Purple", label='Benchmark')
        classificationStrategy_portval_inSample.plot(ax=ax, color="Green", label='Strategy Learner')

        plt.title("Manual Strategy Vs Strategy Learner Vs Benchmark - In sample")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normed Portfolio Value")
        plt.legend()
        plt.savefig("./images/experiment1_In Sample.png")
        plt.close()

        ax = trades_outOfSample_manual.plot(fontsize=12, color="red", label="Manual Strategy")
        trades_outOfSample_bench.plot(ax=ax, color="Purple", label='Benchmark')
        classificationStrategy_portval_outOfSample.plot(ax=ax, color="Green", label='Strategy Learner')

        plt.title("Manual Strategy Vs Strategy Learner Vs Benchmark - Out of sample")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normed Portfolio Value")
        plt.legend()
        plt.savefig("./images/experiment1_Out Of Sample.png")
        plt.close()
