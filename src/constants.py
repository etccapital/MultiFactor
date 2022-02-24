import src.dataloader as dl
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
START_DATE = '20110101'
END_DATE = '20201231'
rebalancing_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='BM')

INDEX_COLS = ['date', 'stock']
FACTORS = {
    'value': ['pb_ratio_ttm', 'pe_ratio_ttm', 'pcf_ratio_ttm']
}
BASIC_INFO_COLS = ['market_value', 'open', 'close']
