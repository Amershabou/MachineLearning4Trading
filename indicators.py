import pandas as pd
from util import get_data
import matplotlib.pyplot as plt
import datetime as dt



class Indicators:

    def __init__(self):
        self.normalizedTypicalPrices = None
        self.typicalPrices = None
        self.normalizedPrices = None
        self.prices = None
        self.sma = None
        self.std = None

    def author(self):
        return "ashabou3"

    def getPrices(self, symbol="JPM", sd=pd.datetime(2008, 1, 1), ed=pd.datetime(2009, 12, 31)):
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates).drop(columns=['SPY'], axis=1)
        highs = get_data([symbol], dates,  colname="High").drop(columns=['SPY'], axis=1)
        lows = get_data([symbol], dates, colname="Low").drop(columns=['SPY'], axis=1)
        typicalPrices = (prices + highs + lows) / 3
        normalizedTypicalPrices = typicalPrices/typicalPrices.ix[0]
        normalizedPrices = prices/prices.ix[0]

        self.typicalPrices = typicalPrices
        self.normalizedTypicalPrices = normalizedTypicalPrices
        self.prices = prices
        self.normalizedPrices = normalizedPrices
        return prices, normalizedPrices

    def getNormalizedPrices(self, symbol="JPM", sd=pd.datetime(2008, 1, 1), ed=pd.datetime(2009, 12, 31)):
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates).drop(columns=['SPY'], axis=1)
        normalizedPrices = prices/prices.ix[0]
        return normalizedPrices
    def getPricesNDaysBack(self, symbol, sd=pd.datetime(2010, 1, 1), ed=pd.datetime(2011, 12, 31)):
        dates = pd.date_range(sd, ed)
        pricesNDays = get_data([symbol], dates).drop(columns=['SPY'], axis=1)
        normalizedPricesNDays = pricesNDays/pricesNDays.ix[0]

        return pricesNDays, normalizedPricesNDays

    def getEMA(self, window, prices):
        return prices.ewm(ignore_na=False,span=window,min_periods=0,adjust=True).mean()

    def calculateSMAIndicator(self, window, symbol="JPM", sd=pd.datetime(2008, 1, 1), ed=pd.datetime(2009, 12, 31)):
        normalized = self.getNormalizedPrices(self, symbol=symbol, sd=sd, ed=ed)
        sma = normalized.rolling(window=window, center=False).mean()
        # self.plotIndicators(self, self.sma, "SMA", "Simple Moving Averages (SMA)", "sma")
        return sma

    def pricesToSma(self, symbol="JPM", sd=pd.datetime(2008, 1, 1), ed=pd.datetime(2009, 12, 31)):
        pToSma = self.normalizedPrices/self.calculateSMAIndicator(self, 20,  symbol=symbol, sd=sd, ed=ed)
        return pToSma
        # self.plotIndicators(self, pToSma, "pricesToSma", "prices To SMA", "pricesToSma")
    def calculateGoldenCrossIndicator(self, symbol="JPM", sd=pd.datetime(2008, 1, 1), ed=pd.datetime(2009, 12, 31)):
        prices200, normalized200 = self.getPricesNDaysBack(self,symbol=symbol, sd=sd - dt.timedelta(days=285),
                                                     ed=ed)
        prices50, normalized50 = self.getPricesNDaysBack(self,symbol=symbol, sd= sd - dt.timedelta(days=72),
                                                     ed=ed)

        sma_200 = normalized200.rolling(window=200, center=False).mean()
        sma_50 = normalized50.rolling(window=50, center=False).mean()
        # self.plotIndicators(self, sma_200, "SMA200", "Golden Cross", "goldenCross", sma_50, "SMA50")
        return sma_200, sma_50

    def calculateGoldenCrossOneVectorIndicator(self, symbol="JPM", sd=pd.datetime(2008, 1, 1), ed=pd.datetime(2009, 12,31)):
        sma_200, sma_50 = self.calculateGoldenCrossIndicator(self, symbol=symbol, sd=sd, ed=ed)
        sma = sma_200/sma_50
        # self.plotIndicators(self, sma, "Golden Cross SMA", "Golden Cross SMA", "goldenCrossOneVector")
        return sma
    def calculateBBPIndicator(self, window, symbol="JPM", sd=pd.datetime(2008, 1, 1), ed=pd.datetime(2009, 12,31)):
        normalizedPrices = self.getNormalizedPrices(self, symbol=symbol, sd=sd, ed=ed)
        std = normalizedPrices.rolling(window=window, center=False).std()
        sma = self.calculateSMAIndicator(self, window=window,symbol=symbol, sd=sd, ed=ed)
        bb_upper = sma + (2 * std)
        bb_lower = sma - (2 * std)
        bb_percentage = (normalizedPrices- bb_lower)/(bb_upper - bb_lower)
        # self.plotIndicators(self,bb_percentage, "BBP", "Bollinger Bands Percentage (BBP)", "bbp")
        return bb_percentage

    def calculateTSIIndicator(self, symbol="JPM", sd=pd.datetime(2007, 12, 1), ed=pd.datetime(2009, 12,31), sample="in"):


        if sample == "Out of":
            sd = pd.datetime(2009, 12, 1)

        prices, normalized = self.getPricesNDaysBack(self, symbol=symbol, sd=sd, ed=ed)


        price_diff = prices - prices.shift(1)
        ema_25 = self.getEMA(self,25, price_diff)
        ema_13 = self.getEMA(self,13, ema_25)

        price_diff_abs = abs(price_diff)
        ema_25_abs = self.getEMA(self,25, price_diff_abs)
        ema_13_abs = self.getEMA(self,13, ema_25_abs)

        tsi = ema_13 / ema_13_abs

        tsi = tsi.truncate(before=pd.datetime(2008, 1, 1))
        # self.plotIndicators(tsi, "TSI", "True Strength Index (TSI)", "tsi")

        return tsi

    def calculateMomentumIndicator(self, window=20, symbol="JPM", sd=pd.datetime(2007, 12, 1), ed=pd.datetime(2009, 12,31)):
        prices, normalized = self.getPricesNDaysBack(self, symbol=symbol, sd=sd, ed=ed)
        momentum = (prices / prices.shift(window) - 1)
        # self.plotIndicators(self,momentum, "Momentum", "Momentum", "momentum")
        return momentum

    def calculateMACDIndicator(self, window=26):
        macd = self.getEMA(window, self.normalizedPrices) - self.getEMA(window, self.normalizedPrices)
        ema9MACD = macd.ewm(ignore_na=False, span=9, min_periods=0, adjust=True).mean()
        self.plotIndicators(self,macd, "MACD", "Moving Average Convergence/Divergence oscillator (MACD)", "macd", ema9MACD, "EMA9MACD")
        return macd, ema9MACD


    def plotIndicators(self, predictor, predictor_label, title, file_name, second_vector=None, second_vector_label=None):
        plt.figure(figsize=(15, 5))
        xdate = [x.date() for x in self.prices.index]
        plt.plot(predictor, label=predictor_label)
        if predictor_label == "SMA":
            plt.plot(self.normalizedPrices, label="Prices")

        if second_vector is not None and second_vector_label is not None:
            plt.plot(second_vector, label=second_vector_label)
        if predictor_label == "Momentum":
            # print(predictor[["JPM"]])
            # print(predictor.index)
            # plt.fill_between(predictor.index, 0, where= predictor > 0.00, color='g',
            #              alpha=0.1)

            plt.axhline(y=0.0, color='r',  xmin=xdate[0], xmax=xdate[-1], linestyle='dashed')
            # plt.fill_between(predictor.index, predictor["JPM"], 0, where=predictor > 0.0, interpolate=True)

        plt.xlabel("Date")
        if second_vector_label:
            plt.ylabel("{} & {}".format(predictor_label, second_vector_label))
        else:
            plt.ylabel(predictor_label)
        plt.legend(loc='best')
        plt.xticks(rotation=30)
        plt.grid()
        plt.title(title)
        plt.savefig("./images/{}.png".format(file_name))
        plt.close()


    def run(self):
        self.getPricesNDaysBack(symbol="JPM", sd=pd.datetime(2007, 12, 1), ed=pd.datetime(2009, 12, 31))
        self.getPrices(symbol="JPM", sd=pd.datetime(2008, 1, 1), ed=pd.datetime(2009, 12, 31))
        self.calculateSMAIndicator(20)
        self.calculateBBPIndicator(20)
        self.CalculateTSIIndicator()
        self.calculateGoldenCrossIndicator()
        self.calculateGoldenCrossOneVectorIndicator()
        self.calculateMomentumIndicator(20)
        self.pricesToSma()
        # self.calculateMACDIndicator()
        plt.figure(figsize=(15, 5))
        plt.plot(self.prices, label="Prices")
        plt.xlabel("Date")
        plt.ylabel("Prices")
        plt.legend(loc='best')
        plt.xticks(rotation=30)
        plt.grid()
        plt.title("Prices")
        plt.savefig("./images/prices.png")
        plt.close()






