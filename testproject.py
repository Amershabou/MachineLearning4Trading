import ManualStrategy as ms
import pandas as dt
import StrategyLearner as sl
import experiment1 as ex1
import experiment2 as ex2


def author():
    return "ashabou3"


if __name__ == "__main__":
    manualStrategy = ms.ManualStrategy()
    sample = "In"

    trades_inSample = manualStrategy.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                                                            ed=dt.datetime(2009, 12, 31), sv=100000, sample=sample)
    sample = "Out of"

    trades_outOfSample = manualStrategy.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1),
                                                            ed=dt.datetime(2011, 12, 31), sv=100000,  sample=sample)

    learner = sl.StrategyLearner(verbose=False, impact=0.0, commission=0.0) # constructor
    learner.add_evidence(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000) # training phase
    df_trades = learner.testPolicy(symbol="JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000) # testing phase

    experiment_1 = ex1.experimnent1()
    experiment_1.runExperiment1()

    experiment_2 = ex2.experimnent2()
    experiment_2.runExperiment2()