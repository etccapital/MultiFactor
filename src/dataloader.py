import rqdatac as rq
import numpy as np
import pandas as pd
import os
import json
import pathos
from src.constants import *
from concurrent.futures import ThreadPoolExecutor
from src.utils import *

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

stock_names = [normalize_code(csv_name.split(".")[0]) for csv_name in csv_names]

def load_stock_info():
    """
        Parrallel computing speeds up the process x10 times

        Important lessons about parallel computing:
        Here we use multiprocessing instead of multithreading since reading in data can be a CPU-bound task i.e. a computationally heavy one
        as opposed to a I/O bound task i.e. tasks whose time are mainly spend on waiting for data to come in/go out.
        Also, since 'get_df' is a local function, the multiprocessing module in the standard library cannot parallelly
        execute it--it can only parallelly map objects which can be pickled/serialized, e.g. functions that are global.
        Even writing 'get_df' and 'load_stock_info' into a single class does not resolve this issue.

        But this can be solved using the pathos library.

        If this function doesn't work for unknown reasons, try calling it for the second time, or restarting jupyter notebook.
        I've tested this function in another notebook--it doesn't work somehow until I restarted vscode, after which it 
        always worked fine. Not sure why this is the case but parallel computing produces weird errors sometimes.
    Returns:
        List[pd.DataFrame]: a list containing many dataframes, each corresponding to a stock
    """
    def get_df(name):
        return pd.read_csv(stock_path+name)
    with pathos.multiprocessing.ProcessPool(pathos.helpers.cpu_count()) as pool:
        stock_info_list = pool.map(get_df, csv_names)
    # with ThreadPoolExecutor() as executor:
        # stock_info_list = executor.map(get_df, csv_names)
    return list(stock_info_list)

@timer
def load_basic_info():
    """
    Returns:
        pd.DataFrame: a dataframe containing the daily information of all stocks on all trading days
    """
    data_path = "./Data/raw_data/"
    file_name = 'df_basic_info.h5'
    if not os.path.exists(data_path + file_name):
        df_basic_info = pd.concat(load_stock_info(), axis=0).rename(columns={'code': 'stock'})
        df_basic_info = df_basic_info.loc[:, df_basic_info.isin(INDEX_COLS + NECESSARY_COLS) ]
        df_basic_info.to_hdf(data_path + file_name, key=file_name)
    df_basic_info = pd.read_hdf(data_path + file_name, key=file_name)
    return df_basic_info
# def load_price_data(col='close'): 
#     # concatenate the price column from each csv
#     results = load_basic_info()
#     price_data = pd.concat([result[col] for result in results], axis=1)
#     stock_names = [dl.normalize_code(csv_name.split(".")[0]) for csv_name in csv_names]
#     price_data.columns = stock_names
#     return price_data

@timer
def load_rebalancing_dates():
    """The rebalancing dates are the last trading date in each month.
    """
    data_folder = os.path.join(DATAPATH, "raw_data")
    file_name = 'rebalancing_dates'
    data_path = os.path.join(data_folder, file_name + ".h5")    
    if not os.path.exists(data_path):
        #load the dataframe with basic information on all trading dates
        df_basic_info = load_basic_info()
        df_basic_info['date'] = pd.to_datetime(df_basic_info['date'])
        df_basic_info['year'] = df_basic_info['date'].dt.year
        df_basic_info['month'] = df_basic_info['date'].dt.month
        df_basic_info['date'] = pd.to_datetime(df_basic_info['date'])
        #groupby year and month first, then take the last date out of each group
        rebalancing_dates = df_basic_info[(df_basic_info['date'] >= START_DATE) & (df_basic_info['date'] <= END_DATE)].groupby(['year', 'month'])['date'].max().values
        pd.Series(rebalancing_dates).to_hdf(data_path, key=file_name)
    rebalancing_dates = pd.to_datetime(pd.read_hdf(data_path).values)
    return rebalancing_dates

def load_industry_mapping():
    if not os.path.exists("./Data/raw_data/industry_mapping.h5"):
        # Extract industry mapping data from ricequant if it's not on the local computer.
        # Extracting from ricequant is quite time consuming. Alternaively, you can download the data from the 
        # cloud folder
        indus_to_stock = {industry: rq.industry(industry) for industry in industry_codes}
        stock_to_indus = {}
        for indus, stock_names in indus_to_stock.items():
            for stock in stock_names:
                if stock in stock_to_indus: print(f"{stock} repeated!")
                stock_to_indus[stock] = indus
        df_indus_mapping = pd.Series(stock_to_indus, name='secon_indus_code').to_frame()
        df_indus_mapping['pri_indus_code'] = df_indus_mapping['secon_indus_code'].str[0]
        df_indus_mapping.to_hdf("./Data/raw_data/industry_mapping.h5", key='industry_mapping')

    # Load the full industry mapping containing industry codes(A to S), industry names in Chinese, and industry names in English of each stock for both primary and secondary industries.
    # The full industry mapping dataframe is obtained by first loading a main dataframe mapping stocks to their industry codes, and then merging the rest two dataframe, which maps industry codes to industry
    # names, onto this dataframe.
    # 'industry_code_to_names.xlsx' is artificially created based on information on https://www.ricequant.com/doc/rqdata/python/stock-mod.html#industry-获取某行业股票列表 
    df_pri_indus_names = pd.read_excel(os.path.join(DATAPATH, 'raw_data', 'industry_code_to_names.xlsx'), 'Primary Industries')
    df_secon_indus_names = pd.read_excel(os.path.join(DATAPATH, 'raw_data', 'industry_code_to_names.xlsx'), 'Secondary Industries')
    df_indus_mapping = pd.read_hdf("./Data/raw_data/industry_mapping.h5", key='industry_mapping').reset_index().rename(columns={'index': 'stock'})
    df_indus_mapping = df_indus_mapping.merge(df_pri_indus_names, how='left', left_on='pri_indus_code', right_on='pri_indus_code' )
    df_indus_mapping = df_indus_mapping.merge(df_secon_indus_names, how='left', left_on='secon_indus_code', right_on='secon_indus_code' )
    df_indus_mapping = df_indus_mapping.set_index('stock')
    # The full industry mapping dataframe should contain at least the following 6 columns
    assert(set(df_indus_mapping.columns).issuperset(
           set(['Primary Industry', 'Secondary Industry', '一级行业', '二级行业', 'pri_indus_code', 'secon_indus_code']) 
           )
    )
    #depending on user input, choose which set of columns to use as industry names
    # if form == 'english':
    #     indus_cols = ['Primary Industry', 'Secondary Industry']
    # elif form == '中文':
    #     indus_cols = ['一级行业', '二级行业']
    # elif form == 'code':
    #     indus_cols = ['pri_indus_code', 'secon_indus_code']
    # else:
    #     raise Exception(f"'{form}' is not a valid input for form!")
    return df_indus_mapping

def load_st_data(stock_names, dates) -> pd.DataFrame:
    """
    stock_names: an iterable of stock names
    returns: a multindex(date and stockname) dataframe indicating whether a stock is an ST stock on a given date
    """
    name = 'is_st'
    #if the dataframe is not stored in the local folder then we fetch it first
    if not os.path.exists('./Data/raw_data/is_st.h5'):
        df_is_st = rq.is_st_stock(stock_names, START_DATE, END_DATE).stack()
        df_is_st.to_hdf('./Data/raw_data/is_st.h5', key=name)
    #load the dataframe
    df_is_st = pd.read_hdf('./Data/raw_data/is_st.h5', key=name).rename(name)
    # df_is_st = df_is_st[df_is_st.index.get_level_values(1).isin(stock_names) & df_is_st.index.get_level_values(0).isin(dates)]
    return df_is_st

def load_suspended_data(stock_names, dates):
    """
    stock_names: an iterable of stock names
    returns: a multindex(date and stockname) dataframe indicating whether a stock is a suspended stock on a given date
    """
    name = 'is_suspended'
    #if the dataframe is not stored in the local folder then we fetch it first
    if not os.path.exists('./Data/raw_data/is_suspended.h5'):
        df_is_suspended = rq.is_suspended(stock_names, START_DATE, END_DATE).stack()
        df_is_suspended.to_hdf('./Data/raw_data/is_suspended.h5', key=name)
    #load the dataframe
    df_is_suspended = pd.read_hdf('./Data/raw_data/is_suspended.h5', key=name).rename(name)
    df_is_suspended = df_is_suspended[df_is_suspended.index.get_level_values(1).isin(stock_names) & df_is_suspended.index.get_level_values(0).isin(dates)]
    return df_is_suspended

def load_listed_dates(selected_stock_names=None):
    #get the listed date for each stock
    #the listed date of a stock is the earliest date in the stock's csv under ./Data/stock_data
    data_path = "./Data/raw_data/"
    file_name = "listed_dates.h5"
    if not os.path.exists(data_path + file_name):
        stock_info_list = load_stock_info()
        listed_dates = {normalize_code(result['code'][0]): result['date'].min() for result in stock_info_list}
        listed_dates = pd.DataFrame(pd.Series(listed_dates), columns=['listed_date']).sort_index().astype('datetime64')
        listed_dates.to_hdf(data_path + file_name, key=file_name)
    listed_dates = pd.read_hdf(data_path + file_name, key=file_name)
    if selected_stock_names is not None:
        listed_dates = listed_dates[listed_dates.index.isin(selected_stock_names)]
    return listed_dates

def load_factor_data(factor: str) -> pd.DataFrame:
    ''' Something something
    '''
    try:
        factor_data = pd.read_hdf(DATAPATH + f'factor/{factor}.h5')
    except:
        print(f'{factor}.h5 not found')
    return factor_data

def load_index_data(index_code='sh000300'):
    # CSI 300 沪深300
    df_index = pd.read_csv(DATAPATH + 'index_data/' + index_code + '.csv',usecols=['date', 'open','close','change'],index_col=['date']).sort_index(ascending=True)
    df_index.columns = ['CSI_300_' + col for col in df_index.columns]
    df_index.index = df_index.index.values.astype('datetime64')
    ((df_index['CSI_300_change'] + 1).cumprod() - 1).plot()
    return df_index

def download_factor_data(stock_names: np.array, factor_name: str, startdate: str, enddate: str) -> None:
    for stock in stock_names:
        if not os.path.exists(f"./Data/factor/{factor_name}.h5"):
            factor_frame = rq.get_factor(stock_names, factor_name, startdate, enddate)
            factor_frame.to_hdf(DATAPATH + f'factor/{factor_name}.h5', key='factor')


