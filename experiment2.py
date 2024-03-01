import ManualStrategy as ms
import pandas as dt
import StrategyLearner as sl
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals


class experimnent2(object):
    def __init__(self):
        pass

    def author(self):
        return "ashabou3"

    def runExperiment2(self):
        impacts = [0.00, 0.005, 0.01, 0.05]
        for impact in impacts:
            learner = sl.StrategyLearner(verbose=False, impact=impact, commission=0.0)  # constructor
            learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                 sv=100000)  # training phase

            df_trades_inSample = learner.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                                    sv=100000)
            classificationStrategy_portval = compute_portvals(df_trades_inSample, learner.prices, "JPM", learner.impact, learner.commission, 10000)
            daily_returns = (classificationStrategy_portval / classificationStrategy_portval.shift(1)) - 1
            daily_returns = daily_returns[1:]
            cr = (classificationStrategy_portval.iloc[-1] / classificationStrategy_portval.iloc[0]) - 1
            adr = daily_returns.mean()
            sddr = daily_returns.std()
            sr = (adr / sddr)

            print("Impact: " + str(impact) + " mean: " + str(adr) + " sddr: " + str(sddr) + " cumulative Return: " + str(cr) + " Sharpe Ratio: " + str(sr))