
from marketsimcode import compute_portvals
from indicators import Indicators
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd


class ManualStrategy(object):
	def __init__(self):
		pass

	def author(self):
		return 'ashabou3'

	def testPolicy(self, symbol,sd,ed, sv, sample="In"):
		prices, normedPrices = Indicators.getPrices(Indicators, symbol, sd, ed)
		action = 0
		momentum = Indicators.calculateMomentumIndicator(Indicators, 20, symbol, sd - dt.timedelta(days=29),ed)
		pricesToSma = Indicators.pricesToSma(Indicators, symbol, sd - dt.timedelta(days=28), ed).dropna()
		bbp = Indicators.calculateBBPIndicator(Indicators, 20, symbol, sd - dt.timedelta(days=28), ed)
		gc = Indicators.calculateGoldenCrossOneVectorIndicator(Indicators, symbol, sd - dt.timedelta(days=2),ed)

		index=0
		buy_date =[]
		sell_date=[]
		df_trades = prices.copy()
		df_trades[:] = 0.0
		dates = df_trades.index

		for i in range(len(dates) - 1):
			if action == 0:
				if bbp.ix[i,symbol] < 0.0 or pricesToSma.ix[i,symbol] < 0.95 or gc.ix[i,symbol] > 1.20 and momentum.ix[i,symbol] > 0.00:
					df_trades.ix[i] = 1000
					action = 1
					index +=1
					buy_date.append(prices.index[i].date())
				elif bbp.ix[i,symbol] > 1.0 or pricesToSma.ix[i,symbol] > 1.05 or gc.ix[i,symbol] < 0.88 and momentum.ix[i,symbol] < 0.00:
					df_trades.ix[i] = -1000
					action=-1
					index+=1
					sell_date.append(prices.index[i].date())
			elif action == -1:
				if bbp.ix[i,symbol] < 0.0 or pricesToSma.ix[i,symbol] < 0.95 or gc.ix[i,symbol] > 1.20 and momentum.ix[i,symbol] > 0.00:
					df_trades.ix[i] = 2000
					action = 1
					index +=1
					buy_date.append(prices.index[i].date())
			elif action == 1:
					if bbp.ix[i,symbol] > 1.0 or pricesToSma.ix[i,symbol] > 1.05 or gc.ix[i,symbol] < 0.80 and momentum.ix[i,symbol] < 0.00:
						df_trades.ix[i] = -2000
						action = -1
						index +=1
						sell_date.append(prices.index[i].date())


		if action==1:
			df_trades.ix[i] = -1000
		if action == -1:
			df_trades.ix[i] = 1000

		benchmark = prices.copy()
		benchmark[symbol] = 0.0
		benchmark.ix[0] = 1000
		benchmark.ix[-1] = -1000

		manualStrategy_portval = compute_portvals(df_trades, prices, symbol, 0.005, 9.95, sv)
		benchmark_portval = compute_portvals(benchmark, prices, symbol, 0.005, 9.95, sv)

		# Manual Strategy metrics
		daily_returns = (manualStrategy_portval / manualStrategy_portval.shift(1)) - 1
		daily_returns = daily_returns[1:]
		cr = (manualStrategy_portval.iloc[-1] / manualStrategy_portval.iloc[0]) - 1
		adr = daily_returns.mean()
		sddr = daily_returns.std()
		# print("Manual Strategy metrics")
		# print(str(cr), str(adr), str(sddr))
		# Benchmark metrics
		benchmark_daily_returns = (benchmark_portval / benchmark_portval.shift(1)) - 1
		benchmark_daily_returns = benchmark_daily_returns[1:]
		cr = (benchmark_portval.iloc[-1] / benchmark_portval.iloc[0]) - 1
		adr = benchmark_daily_returns.mean()
		sddr = benchmark_daily_returns.std()
		# print("Benchmark metrics")
		# print(str(cr), str(adr), str(sddr))



		normed_manualStrategy_portval = manualStrategy_portval / manualStrategy_portval[0]
		normed_benchmark_portval = benchmark_portval / benchmark_portval[0]
		ax = normed_manualStrategy_portval.plot(fontsize=12, color="red", label="Manual Strategy")
		normed_benchmark_portval.plot(ax=ax, color="Purple", label='Benchmark')
		plt.ylim([0.5, 2.0])
		for date in buy_date:
			y = normed_manualStrategy_portval[date]/2.5
			ax.axvline(date, color="blue", linewidth=0.5, ymin=0, ymax=y,linestyle="--")
		for date in sell_date:
			y = normed_manualStrategy_portval[date] / 2.5
			ax.axvline(date, color="black",  linewidth=0.5, ymin=y, ymax=1,linestyle="--")
		plt.title(" Manual Strategy - " + sample +" sample.")
		ax.set_xlabel("Date")
		ax.set_ylabel("Normed Value")
		plt.legend()
		plt.savefig("./images/{} sample.png".format(sample))
		plt.close()

		return normed_manualStrategy_portval, normed_benchmark_portval
