import os
import pandas as pd

DATAPATH = './data/' #seperating raw and processed data
stock_path = DATAPATH + 'stock_data/'
csv_names = os.listdir(path=stock_path)
industry_codes = [f'A0{i}' for i in range(1, 6)] + [f'B0{i}' for i in range(6, 10)] + [f'B{i}' for i in range(10, 13)] \
    + [f'C{i}' for i in range(13, 44)] + [f'D{i}' for i in range(44, 47)] + [f'E{i}' for i in range(47, 51)] \
    + ['F51', 'F52'] + [f'G{i}' for i in range(53, 61)] + ['H61', 'H62'] + [f'I{i}' for i in range(63, 66)] + \
    [f'J{i}' for i in range(66, 70)] + ['K70'] + ['L71', 'L72'] + [f'M{i}' for i in range(73, 76)] + \
    [f'N{i}' for i in range(76, 79)] + [f'O{i}' for i in range(79, 82)] + ['P82'] + ['Q83', 'Q84'] + \
    [f'R{i}' for i in range(85, 90)] + ['S90']

#backtesting timeframe
START_DATE = '2011-01-01'
END_DATE = '2020-12-31'
REBALANCING_DATES = pd.to_datetime(pd.read_hdf(os.path.join(DATAPATH, 'raw_data', 'rebalancing_dates.h5')).values)
#Ensure that there is a rebalancing date on every month i.e. the time window between two rebalancing dates can be
#no longer than 40 days
assert( ((REBALANCING_DATES[1:] - REBALANCING_DATES[:-1]) > pd.Timedelta('40d') ).sum() == 0)

INDEX_COLS = ['date', 'stock']
PRIMARY_INDUSTRY_COL = '一级行业'
SECONDARY_INDUSTRY_COL = '二级行业'
INDUSTRY_COLS = [PRIMARY_INDUSTRY_COL, SECONDARY_INDUSTRY_COL]
NECESSARY_COLS = ['market_value', 'open', 'close', 'next_period_return', ] + INDUSTRY_COLS
