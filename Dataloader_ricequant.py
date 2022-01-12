import rqdatac as rq
import numpy as np
import pandas as pd
import os
import json
import pathos
from constants import *

#TODO: Refactor this into a class if needed in the future

# Use rq_crendential.json to fill out Ricequant credentials
# WARNING: MAKE SURE rq_crendential.json ARE NOT COMMITTED TO GITHUB
CRED_FILE = './rq_credential.json'
with open(CRED_FILE) as file:
    rq_cred = json.load(file)

RQ_USER, RQ_PASS = rq_cred['user'], rq_cred['password']

def rq_initialize(): 
    rq.init(RQ_USER, RQ_PASS)

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

    symbol = symbol.upper()
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

stock_names = [dl.normalize_code(csv_name.split(".")[0]) for csv_name in csv_names]

def load_basic_info():
    #parrallel computing speeds up the process x10 times
    #returns a list containing many dataframes, each corresponding to a stock

    def get_df(name):
        return pd.read_csv(stock_path+name)
        # return pd.read_csv(stock_path+name).set_index(['date'])

    with pathos.multiprocessing.ProcessPool(pathos.helpers.cpu_count()) as pool:
        results = pool.map(get_df, csv_names)
    return results

def load_price_data(col='close'): 
    #concatenate the price column from each csv
    results = load_basic_info()
    price_data = pd.concat([result[col] for result in results], axis=1)
    stock_names = [dl.normalize_code(csv_name.split(".")[0]) for csv_name in csv_names]
    price_data.columns = stock_names
    return price_data

def load_industry_mapping():
    if not os.path.exists("./Data/raw_data/industry_mapping.h5"):
        indus_to_stock = {industry: rq.industry(industry) for industry in industry_codes}
        stock_to_indus = {}
        for indus, stock_names in indus_to_stock.items():
            for stock in stock_names:
                if stock in stock_to_indus: print(f"{stock} repeated!")
                stock_to_indus[stock] = indus
        df_indus_mapping = pd.Series(stock_to_indus, name='secon_indus_code').to_frame()
        df_indus_mapping['pri_indus_code'] = df_indus_mapping['secon_indus_code'].str[0]
        df_indus_mapping.to_hdf("./Data/raw_data/industry_mapping.h5", key='industry_mapping')
    df_indus_mapping = pd.read_hdf("./Data/raw_data/industry_mapping.h5", key='industry_mapping')
    return df_indus_mapping

def load_st_data(stock_names) -> pd.DataFrame:
    """
    stock_names: an iterable of stock names
    returns: a multindex(date and stockname) dataframe indicating whether a stock is an ST stock on a given date
    """
    #if the dataframe is not stored in the local folder then we fetch it first
    if not os.path.exists('./Data/raw_data/is_st.h5'):
        df_is_st = rq.is_st_stock(stock_names, START_DATE, END_DATE).stack()
        df_is_st.to_hdf('./Data/raw_data/is_st.h5', key='is_st')
    #load the dataframe
    df_is_st = pd.read_hdf('./Data/raw_data/is_st.h5', key='is_st')
    df_is_st = df_is_st[df_is_st.index.get_level_values(1).isin(stock_names)]
    return df_is_st

def load_suspended_data(stock_names):
    """
    stock_names: an iterable of stock names
    returns: a multindex(date and stockname) dataframe indicating whether a stock is a suspended stock on a given date
    """
    #if the dataframe is not stored in the local folder then we fetch it first
    if not os.path.exists('./Data/raw_data/is_suspended.h5'):
        df_is_suspended = rq.is_suspended_stock(stock_names, START_DATE, END_DATE).stack()
        df_is_suspended.to_hdf('./Data/raw_data/is_suspended.h5', key='is_suspended')
    #load the dataframe
    df_is_suspended = pd.read_hdf('./Data/raw_data/is_suspended.h5', key='is_suspended')
    df_is_suspended = df_is_suspended[df_is_suspended.index.get_level_values(1).isin(stock_names)]
    return df_is_suspended

def load_listed_dates():
    #get the listed date for each stock
    #the listed date of a stock is the earliest date in the stock's csv under ./Data/stock_data
    if not os.path.exists("./Data/raw_data/listed_dates"):
        results = load_basic_info()
        listed_dates = [result.index.min() for result in results]
        listed_dates = pd.Series(dict(zip(stock_names, listed_dates)))
        listed_dates.to_hdf("./Data/raw_data/listed_dates", key='listed_dates')
    listed_dates = pd.read_hdf("./Data/raw_data/listed_dates", key='listed_dates')
    return listed_dates

def load_factor_data(factor: str) -> pd.DataFrame:
    ''' Something something
    '''
    try:
        factor_data = pd.read_hdf(DATAPATH + f'factor/{factor}.h5')
    except:
        print(f'{factor}.h5 not found')
            
    return factor_data

def download_factor_data(stock_names: np.array, factor_name: str, startdate: str, enddate: str) -> None:
    if not os.path.exists(f"./Data/factor/{factor_name}.h5"):
        factor_frame = rq.get_factor(stock_names, factor_name, startdate, enddate)
        factor_frame.to_hdf(DATAPATH + f'factor/{factor_name}.h5', key='factor')

