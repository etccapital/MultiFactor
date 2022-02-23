import Dataloader_ricequant as dl
import os

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

INDEX_COLS = ['date', 'stock']
FACTORS = {
    'value': ['pb_ratio_ttm', 'pe_ratio_ttm', 'pcf_ratio_ttm']
}
TEST_FACTORS = ['PE_TTM', 'PS_TTM', 'PC_TTM', 'PB'] + FACTORS['value']
BASIC_INFO_COLS = ['market_value', 'open', 'close']
RQ_FACTORS = ['pb_ratio_ttm', 
                'return_on_equity_ttm', 
                'market_cap_3', 
                'dividend_yield_ttm', 
                'book_to_market_ratio_ttm', 
                'VOL3']