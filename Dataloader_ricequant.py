import rqdatac as rq
import numpy as np
import pandas as pd
import os
import json

# Use rq_crendential.json to fill out Ricequant credentials
# WARNING: MAKE SURE rq_crendential.json ARE NOT COMMITTED TO GITHUB
CRED_FILE = './rq_credential.json'
with open(CRED_FILE) as file:
    rq_cred = json.load(file)

RQ_USER, RQ_PASS = rq_cred['user'], rq_cred['password']
DATAPATH = './data/'

def rq_initialize():
    rq.init(RQ_USER, RQ_PASS)

def load_price_data():
    stock_path = DATAPATH + 'stock_data/'
    stock_names = os.listdir(path=stock_path)
    initial = pd.read_csv(stock_path + stock_names[0]).set_index('date')
    initial = initial[['close']].rename({'close':initial.iloc[0,0]},axis=1)
    price_data = initial

    for name in stock_names[1:]:
        stock_data = pd.read_csv(stock_path+name).set_index(['date'])
        close_price = stock_data[['close']].rename({'close':stock_data.iloc[0,0]}, axis=1)
        price_data = price_data.join(close_price)

    return price_data

def load_factor_data(factor: str) -> pd.DataFrame:
    ''' Something something

    '''
    try:
        factor_data = pd.read_hdf(DATAPATH + f'factor/{factor}.h5')
    except:
        print(f'{factor}.h5 not found')
            
    return factor_data

def download_factor_data(stock_name: np.array, factor_name: str, startdate: str, enddate: str) -> None:
    factor_frame = rq.get_factor(stock_name, factor_name, startdate, enddate)
    factor_frame.to_hdf(DATAPATH + f'factor/{factor_name}.h5', key='factor')

def normalize_code(symbol, pre_close=None):
    """
    归一化证券代码

    :param code 如000001
    :return 证券代码的全称 如000001.XSHE
    """
    XSHG = 'XSHG'
    SSE = 'XSHG'
    SH = 'XSHG'
    XSHE = 'XSHE'
    SZ = 'XSHE'
    SZE = 'XSHE'
    if (not isinstance(symbol, str)):
        return symbol

    if (symbol.startswith('SZ') and (len(symbol) == 8)):
        ret_normalize_code = '{}.{}'.format(symbol[2:8],SZ)
    elif (symbol.startswith('SH') and (len(symbol) == 8)):
        ret_normalize_code = '{}.{}'.format(symbol[2:8], SH)
    elif (symbol.startswith('00') and (len(symbol) == 6)):
        if ((pre_close is not None) and (pre_close > 2000)):
            # 推断是上证指数
            ret_normalize_code = '{}.{}'.format(symbol, SH)
        else:
            ret_normalize_code = '{}.{}'.format(symbol, SZ)
    elif ((symbol.startswith('399') or symbol.startswith('159') or \
        symbol.startswith('150')) and (len(symbol) == 6)):
        ret_normalize_code = '{}.{}'.format(symbol, SH)
    elif ((len(symbol) == 6) and (symbol.startswith('399') or \
        symbol.startswith('159') or symbol.startswith('150') or \
        symbol.startswith('16') or symbol.startswith('184801') or \
        symbol.startswith('201872'))):
        ret_normalize_code = '{}.{}'.format(symbol, SZ)
    elif ((len(symbol) == 6) and (symbol.startswith('50') or \
        symbol.startswith('51') or symbol.startswith('60') or \
        symbol.startswith('688') or symbol.startswith('900') or \
        (symbol == '751038'))):
        ret_normalize_code = '{}.{}'.format(symbol, SH)
    elif ((len(symbol) == 6) and (symbol[:3] in ['000', '001', '002', 
                                                 '200', '300'])):
        ret_normalize_code = '{}.{}'.format(symbol, SZ)
    else:
        print(symbol)
        ret_normalize_code = symbol

    return ret_normalize_code
