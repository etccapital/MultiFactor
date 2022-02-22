from constants import *
from utils import *
import pandas as pd

def time_and_list_status_preprocess(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function filters original dataframe base on time and listing status

    step 0: Read all csv's and concatenate the desired column from each dataframe
    step 1: Filter out data before START_DATE and after END_DATE(backtesting period) from the raw stock data
    step 2: Filter listed stocks
    step 3: Filter out ST stocks, suspended stocks and stocks that are listed only within one year
    """
    # step 0
    df_backtest = pd.concat(original_df, axis=0).rename(columns={'code': 'stock'}).loc[:, INDEX_COLS + BASIC_INFO_COLS]
    df_backtest['date'] = pd.to_datetime(df_backtest['date'])

    # step 1
    df_backtest = df_backtest[ (START_DATE <= df_backtest['date']) & (df_backtest['date'] <= END_DATE) ]
    df_backtest['stock'] = df_backtest['stock'].apply(lambda stock: dl.normalize_code(stock))

    # have a (date_stock) multi-index dataframe 
    df_backtest = df_backtest.set_index(INDEX_COLS).sort_index()
    df_backtest = df_backtest.unstack(level=1).stack(dropna=False)

    # check stock index
    stock_names = df_backtest.index.get_level_values(1).unique()
    
    # step 2
    # get the listed date
    listed_dates = {dl.normalize_code(result['code'][0]): result['date'].min() for result in original_df}
    listed_dates = pd.DataFrame(pd.Series(listed_dates), columns=['listed_date']).sort_index().astype('datetime64')
    # left join with dataframe 'listed_dates'
    df_backtest = df_backtest.merge(listed_dates, left_on = 'stock', right_index=True, how='left')
    # create a new variable called 'is_listed' to check if a certain stock is listed at that given date
    df_backtest['is_listed_for_one_year'] = (df_backtest.index.get_level_values(level=0).values - df_backtest['listed_date'].values >= pd.Timedelta('1y'))

    # number of non-listed stocks along the time
    non_listed = df_backtest[~df_backtest['is_listed_for_one_year']]
    # num_nonlisted_stock = non_listed.groupby(level=0).count()['is_listed_for_one_year']
    # num_nonlisted_stock.plot.line()

    # load st/suspend data from Ricequant
    df_is_st = dl.load_st_data(stock_names)
    df_is_suspended = dl.load_suspended_data(stock_names)

    # step 3
    # create is_st and is_suspended columns
    df_backtest['is_st'] = df_is_st.values
    df_backtest['is_suspended'] = df_is_suspended.values

    # filter out stocks that are listed within a year, ST, 
    # and suspended stocks, filter data by the stock's listed date
    df_backtest = df_backtest.loc[ (~df_backtest['is_st']) & (~df_backtest['is_suspended']) & (df_backtest['is_listed_for_one_year']), BASIC_INFO_COLS]

    # keep data only on the rebalancing dates
    rebalancing_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='BM')
    df_backtest = df_backtest[df_backtest.index.get_level_values(0).isin(rebalancing_dates)]

    # the current rebalancing date is the last trading day of the current period
    # 'next_period_open' is defined as the stock's open price on the next relancing date
    # 'next_period_return' is the generated return by holding a stock from EOD of current rebalancing date to the start of the next rebalancing date
    df_backtest['next_period_open'] = df_backtest['open'].groupby(level=1).shift(-1).values
    df_backtest['next_period_return'] = (df_backtest['next_period_open'].values - df_backtest['close'].values) / df_backtest['close'].values
    df_backtest = df_backtest[df_backtest.index.get_level_values(0) != df_backtest.index.get_level_values(0).max()]

    assert(df_backtest is not None)
    return df_backtest

def standardization_and_outlier_missing_val_preprocess(df):
    """
    This function filters dataframe with the following steps

    step 1: Replace Outliers with the corresponding threshold
    step 2: Standardization - Subtract mean and divide by std
    step 3: Fill missing factor values with 0
    step 4: Filter out entries with missing return values
    """

    # step 1
    df[TEST_FACTORS] = applyParallel(df[TEST_FACTORS].groupby(level=0), remove_outlier).values

    # step 2
    df[TEST_FACTORS] = applyParallel(df[TEST_FACTORS].groupby(level=0), standardize).values

    # step 3
    df[TEST_FACTORS] = df[TEST_FACTORS].fillna(0).values

    # step 4 
    df = df[df['next_period_return'].notnull() & df['market_value'].notnull()]

    assert(df is not None)
    return df
