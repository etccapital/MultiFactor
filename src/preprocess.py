from src.constants import *
from src.utils import *
import pandas as pd
import src.dataloader as dl
import matplotlib.pyplot as plt

class TimeAndStockFilter:
    """
    This function filters original dataframe base on dates and stocks
    1) Filter out data before START_DATE and after END_DATE(backtesting period) from the raw stock data. 

    - 剔除不在回测区间内的股票信息

    2) Filter listed stocks

    - 选出回测区间内每只股票上市的时间。这一步是为了步骤3，因为在每个选股日筛选ST或者停牌股的前提是股票在该选股日已上市。

    3) Filter out ST stocks, suspended stocks and stocks that are listed only within one year
    - 剔除ST，停牌和次新股（上市未满一年的股票）
    """
    @timer
    def __init__(self, df_basic_info, ):
        self.df_backtest = df_basic_info.copy()

    @timer
    def preprocess(self, ):
        """
        step 0: preprocess the dataframe into desired format
        """
        # turn the date column's data type from str to pd.datetime
        self.df_backtest['date'] = pd.to_datetime(self.df_backtest['date'] )
        # make the # of stocks equal on each date
        # self.df_backtest = self.df_backtest.unstack(level=1).stack(dropna=False)

    @timer
    def filter_dates(self, rebalancing_dates=rebalancing_dates):
        """
        step 1: filter data on rebalancing dates. No need to filter using the backtesting start_date and end_date anymore because 
        """
        # rebalancing_dates contains only dates between start_date and end_date.
        self.df_backtest = self.df_backtest[self.df_backtest['date'].isin(rebalancing_dates)]
        # Filter out data before START_DATE and after END_DATE(backtesting period) from the raw stock data
        # self.df_backtest = self.df_backtest[ (start <= self.df_backtest['date']) & (self.df_backtest['date'] <= end) ]
        self.df_backtest = self.df_backtest.sort_values(by=INDEX_COLS)
        # normalize the stock codes
        self.df_backtest['stock'] = self.df_backtest['stock'].apply(dl.normalize_code)

    @timer
    def filter_stocks(self, visualize=False):
        """
        step 2: Filter out ST stocks, suspended stocks and stocks that are listed only within one year
        """
        #get the stock names and rebalancing dates
        stock_names = self.df_backtest['stock'].unique()
        dates = self.df_backtest['date'].unique()

        # load st/suspend/listed date data from local computer
        df_is_st = dl.load_st_data(stock_names, dates)
        df_is_suspended = dl.load_suspended_data(stock_names, dates)
        df_listed_dates = dl.load_listed_dates(stock_names)

        # create is_st, is_suspended and listed date columns
        self.df_backtest = self.df_backtest.merge(df_is_st, how='left', left_on=INDEX_COLS, right_index=True)
        self.df_backtest = self.df_backtest.merge(df_is_suspended, how='left', left_on=INDEX_COLS, right_index=True)
        self.df_backtest = self.df_backtest.merge(df_listed_dates, how='left', left_on = 'stock', right_index=True)
        # create a new variable called 'is_listed_for_one_year' to check if a certain stock is listed for at least one year at that given date
        self.df_backtest['is_listed_for_one_year'] = (self.df_backtest['date'] - self.df_backtest['listed_date'].values >= pd.Timedelta('1y'))

        # filter out stocks that are listed within a year, ST, and suspended stocks, filter data by the stock's listed date
        self.df_backtest = self.df_backtest.loc[ (~self.df_backtest['is_st']) & (~self.df_backtest['is_suspended']) & (self.df_backtest['is_listed_for_one_year']), :]

        # number of non-listed stocks along the time
        if visualize:
            non_listed = self.df_backtest[~self.df_backtest['is_listed_for_one_year']]
            num_nonlisted_stock = non_listed.groupby(level=0).count()['is_listed_for_one_year']
            num_nonlisted_stock.plot.line()
            plt.show()
    
    def postprocess(self):
        """
        step 3: postprocess the dataframe into desired format
        """
        # the rebalancing date is the last trading day of the period
        # 'next_period_return' is the generated return by holding a stock from end of current rebalancing date to the start of the next rebalancing date
        self.df_backtest['next_period_return'] = (self.df_backtest.groupby('date')['open'].shift(-1).values - self.df_backtest['close'].values) / self.df_backtest['close'].values
        #drop the last period since its 'next_period_return' cannot be calculated
        self.df_backtest = self.df_backtest[self.df_backtest['date'] != self.df_backtest['date'].max()]
        # have a (date, stock) multi-index dataframe
        self.df_backtest = self.df_backtest.set_index(INDEX_COLS)
        self.df_backtest = self.df_backtest[BASIC_INFO_COLS + ['next_period_return']]

    def run(self):
        self.preprocess()
        self.filter_dates()
        self.filter_stocks()
        self.postprocess()
        assert(self.df_backtest is not None)
        return self.df_backtest

@timer
def standardize(df, factors):
    """
    This function filters dataframe with the following steps

    step 1: Replace Outliers with the corresponding threshold
    step 2: Standardization - Subtract mean and divide by std
    step 3: Fill missing factor values with 0
    step 4: Filter out entries with missing return values
    """

    assert(factors is not None)

    # step 1
    df[factors] = applyParallel(df[factors].groupby(level=0), remove_outlier).values

    # step 2
    df[factors] = applyParallel(df[factors].groupby(level=0), standardize).values

    # step 3
    df[factors] = df[factors].fillna(0).values

    # step 4 
    df = df[df['next_period_return'].notnull() & df['market_value'].notnull()]

    assert(df is not None)
    return df
