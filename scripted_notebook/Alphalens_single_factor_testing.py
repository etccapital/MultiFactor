# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os 
os.chdir('../')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from datetime import datetime, timedelta

import rqdatac as rq
import alphalens as al

# %% [markdown]
#
# ## Single Factor Testing using Alphalens

# %% [markdown]
# #### Sector Classification

# %%
# MSCI全球标准行业分类
sectors = ['Energy','Materials','ConsumerDiscretionary','ConsumerStaples',
          'HealthCare','Financials','RealEstate','InformationTechnology',
          'TelecommunicationServices','Utilities','Industrials']
sector_names = {'Energy':'能源','Materials':'原材料','ConsumerDiscretionary':'非必须消费品',
               'ConsumerStaples':'必需消费品','HealthCare':'医疗保健','Financials':'金融',
               'RealEstate':'房地产','InformationTechnology':'信息技术',
                'TelecommunicationServices':'电信服务','Utilities':'公共服务',
                'Industrials':'工业'}

sector_dict = {sector:rq.sector(sector) for sector in sectors}
code_dict = {}
for k, v in sector_dict.items():
    for code in v:
        code_dict[code] = k

# %% [markdown]
# #### Import price data and factor data

# %%
backtest_price_data = pd.read_hdf('10-year non-ST price data.h5')
backtest_price_data = backtest_price_data.fillna(method='ffill')

# %%

# %%
roe = pd.read_hdf('roe.h5').reset_index().set_index(['date','order_book_id']).fillna(method='ffill')

# %%
pb = pd.read_hdf('pb.h5').reset_index().set_index(['date','order_book_id']).fillna(method='ffill')

# %%
mkt_cap = pd.read_hdf('mkt_cap.h5').reset_index().set_index(['date','order_book_id']).fillna(method='ffill')

# %%
vol3 = pd.read_hdf('vol3.h5').reset_index().set_index(['date','order_book_id']).fillna(method='ffill')

# %%
stock_codes = list(set(roe.index.get_level_values(1)))
backtest_price_data = backtest_price_data[stock_codes]
ticker_sector = {key: value for key, value in code_dict.items() if key in stock_codes}

# %% [markdown]
#

# %%
pb_factor_data = al.utils.get_clean_factor_and_forward_returns(factor=pb,
                                                               prices=backtest_price_data,
                                                               periods=(1,5,10),
                                                               max_loss=0.4,
                                                               groupby_labels=sector_names,
                                                               groupby=ticker_sector)

# %%
roe_factor_data = al.utils.get_clean_factor_and_forward_returns(factor=roe,
                                                               prices=backtest_price_data,
                                                               periods=(1,5,10), max_loss=0.4)

# %%
mkt_cap_factor_data = al.utils.get_clean_factor_and_forward_returns(factor=mkt_cap,
                                                                   prices=backtest_price_data,
                                                                   periods=(1,5,10), max_loss=0.4)

# %%
vol3_factor_data = al.utils.get_clean_factor_and_forward_returns(factor=vol3,
                                                                   prices=backtest_price_data,
                                                                   periods=(1,5,10), max_loss=0.4)

# %%
al.tears.create_returns_tear_sheet(factor_data=roe_factor_data)

# %%
al.tears.create_returns_tear_sheet(factor_data=mkt_cap_factor_data)

# %%
al.tears.create_returns_tear_sheet(factor_data=vol3_factor_data)

# %%
al.tears.create_information_tear_sheet(factor_data=roe_factor_data)

# %%
al.tears.create_turnover_tear_sheet(factor_data=roe_factor_data,
                                   turnover_periods=['1D','5D','10D'])

# %%
al.tears.create_returns_tear_sheet(factor_data=pb_factor_data)

# %%
al.tears.create_information_tear_sheet(factor_data=pb_factor_data)

# %%
al.tears.create_turnover_tear_sheet(factor_data=pb_factor_data,turnover_periods=['1D','5D','10D'])

# %% [markdown]
#
