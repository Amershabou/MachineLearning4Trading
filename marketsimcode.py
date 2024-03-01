
  		  	   		  		 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		  		 		  		  		    	 		 		   		 		  
import os
import numpy as np
  		  	   		  		 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  

def author():
    return "ashabou3"
def compute_portvals(
    orders_df,
    prices,
    symbol,
    impact,
    commission,
    start_val=1000000
):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		  		 		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		  		 		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		  		 		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		  		 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		  		 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    # this is the function the autograder will call to test your code  		  	   		  		 		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		  		 		  		  		    	 		 		   		 		  
    # code should work correctly with either input  		  	   		  		 		  		  		    	 		 		   		 		  

    # prices = get_data(symbols, pd.date_range(start_date, end_date)).drop(columns=['SPY'], axis=1)
    prices["Cash"] = 1
    trades = prices.copy()
    trades[trades.columns] = 0.00
    # print(orders_df)
    # print(prices)
    trans_cost = 0.00
    for index, row in orders_df.iterrows():

        trades.ix[index][symbol] += row[0]
        trans_cost += (prices.ix[index][symbol] * row[0] * impact) + commission
        trades.ix[index]["Cash"] += row[0] * prices.ix[index][symbol] * -1


    # print(trades)
    initial_value = start_val - trans_cost
    holdings = trades.copy()
    first_row = trades.ix[0]
    for index, row in holdings.iterrows():
        if index == trades.index[0]:
            row["Cash"] = trades.ix[index]["Cash"] + initial_value
            first_row = holdings.ix[index]
        else:
            holdings.ix[index] = holdings.ix[index] + first_row
            first_row = holdings.ix[index]


    values = prices * holdings
    portvals = values.sum(axis=1)

    return portvals
