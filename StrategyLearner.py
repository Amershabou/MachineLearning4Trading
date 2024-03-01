
import datetime as dt  		  	   		  		 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals
from indicators import Indicators
import RTLearner as rtl
import BagLearner as bl
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
import util as ut
import numpy as np


class StrategyLearner(object):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  		 		  		  		    	 		 		   		 		  
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = None
        self.train_y = None
        self.prices = None

    def author(self):
        return "ashabou3"

    # this method should create a QLearner, and train it for trading
    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=100000,
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """

        # Training on the in sample data and using Classification Random Forest Learner
        prices, normedPrices = Indicators.getPrices(Indicators, symbol, sd, ed)
        self.prices = prices
        momentum = Indicators.calculateMomentumIndicator(Indicators, 20, symbol, sd - dt.timedelta(days=29), ed).dropna()
        pricesToSma = Indicators.pricesToSma(Indicators, symbol, sd - dt.timedelta(days=28), ed).dropna()
        bbp = Indicators.calculateBBPIndicator(Indicators, 20, symbol, sd - dt.timedelta(days=28), ed).dropna()
        gc = Indicators.calculateGoldenCrossOneVectorIndicator(Indicators, symbol, sd - dt.timedelta(days=2), ed).dropna()
        train_x = np.concatenate([momentum, bbp, pricesToSma, gc], axis=1)
        dates = pd.date_range(sd,  dt.datetime(2010, 2, 5))
        prices_all = ut.get_data([symbol], dates).drop(columns=['SPY'], axis=1)  # automatically adds SPY

        train_y = prices.copy().to_numpy()

        for i in range(len(train_y)):
            ret = (prices_all.ix[i + 5, symbol] / prices_all.ix[i, symbol]) - 1
            trade_costs = (prices_all.ix[i, symbol] * self.impact)/prices_all.ix[i, symbol] + self.commission
            ret = ret - trade_costs
            if ret > 0.015:
                train_y[i] = 1
            elif ret < -0.015:
                train_y[i] = -1
            else:
                train_y[i] = 0
        self.train_y = train_y
        learner = bl.BagLearner(rtl.RTLearner, {"leaf_size": 20, "verbose": False}, 20)
        learner.add_evidence(train_x, train_y)
        self.learner = learner


    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=10000,
    ):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		  		 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		  		 		  		  		    	 		 		   		 		  
        """
        # evaluate in sample and out of sample
        prices, normedPrices = Indicators.getPrices(Indicators, symbol, sd, ed)
        momentum = Indicators.calculateMomentumIndicator(Indicators, 20, symbol, sd - dt.timedelta(days=29), ed).dropna()
        pricesToSma = Indicators.pricesToSma(Indicators, symbol, sd - dt.timedelta(days=28), ed).dropna()
        bbp = Indicators.calculateBBPIndicator(Indicators, 20, symbol, sd - dt.timedelta(days=28), ed).dropna()
        gc = Indicators.calculateGoldenCrossOneVectorIndicator(Indicators, symbol, sd - dt.timedelta(days=2), ed).dropna()
        train_x = np.concatenate([momentum, pricesToSma, bbp, gc], axis=1)

        pred_y = self.learner.query(train_x)  # get the predictions

        index = 0
        buy_date = []
        sell_date = []
        df_trades = prices.copy()
        df_trades[:] = 0
        df_trades.ix[0] = 0
        action = 0
        cur = 0

        for i in range(len(pred_y) - 1):
            if action == 0:
                if pred_y[i] == 1:
                    if cur == -1000:
                        df_trades.ix[i] = 2000
                    elif cur == 0:
                        df_trades.ix[i] = 1000
                    else:
                        df_trades.ix[i] = 0
                    cur = 1000
                    action = 1
                    index += 1
                    buy_date.append(prices.index[i].date())
                elif pred_y[i] == -1:
                    if cur == -1000:
                        df_trades.ix[i] = 0
                    elif cur == 0:
                        df_trades.ix[i] = -1000
                    else:
                        df_trades.ix[i] = -2000
                    cur = -1000
                    action = -1
                    index += 1
                    sell_date.append(prices.index[i].date())
                else:
                    df_trades.ix[i] = 0
                    index += 1
            elif action == -1:
                if pred_y[i] == 1:
                    if cur == -1000:
                        df_trades.ix[i] = 2000
                    elif cur == 0:
                        df_trades.ix[i] = 1000
                    else:
                        df_trades.ix[i] = 0
                    cur = 1000
                    action = 1
                    index += 1
                    buy_date.append(prices.index[i].date())
                else:
                    df_trades.ix[i] = 0
                    index += 1
            elif action == 1:
                if pred_y[i] == -1:
                    if cur == -1000:
                        df_trades.ix[i] = 0
                    elif cur == 0:
                        df_trades.ix[i] = -1000
                    else:
                        df_trades.ix[i] = -2000
                    cur = -1000
                    action = -1
                    index += 1
                    sell_date.append(prices.index[i].date())
                else:
                    df_trades.ix[i] = 0
                    index += 1

        if cur <= -1000:
            df_trades.ix[-1] = 1000
        elif cur >= 1000:
            df_trades.ix[-1] = -1000

        # classificationStrategy_portval = compute_portvals(df_trades, prices, symbol, self.impact, self.commission, sv)
        # classificationStrategy_portval = classificationStrategy_portval / classificationStrategy_portval[0]
        # ax = classificationStrategy_portval.plot(fontsize=12, color="red", label="Classification Strategy Learner")
        # plt.title("Classification Strategy Learner - In sample.")
        # ax.set_xlabel("Date")
        # ax.set_ylabel("Normed Portfolio Value")
        # plt.legend()
        # plt.savefig("./images/ClassificationStrategyLearner.png")
        # plt.close()

        return df_trades


