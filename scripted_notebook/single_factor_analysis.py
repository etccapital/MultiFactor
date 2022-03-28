# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: multifactor
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Set up
# Run this section before anything else

# %%
import os 
os.chdir('../')
from src.constants import *

# %%
import os 
from datetime import datetime, timedelta
import src.dataloader as dl
import pandas as pd
import rqdatac as rq
from src.constants import *
from src.utils import *
from src.preprocess import *
from src.single_factor import *

import numpy as np
import scipy
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# %%
dl.rq_initialize()

# %%
df_basic_info = dl.load_basic_info()
filter = TimeAndStockFilter(df_basic_info)
df_backtest = filter.run()

# %% [markdown]
# ### Single Factor Analysis

# %% [markdown]
# #### Financial Quality Factors

# %%
# DOTO: design a function in utils that automatically read all the file names given a folder
QUALITY_FACTORS = ['debt_to_asset_ratio_ttm', 'fixed_asset_ratio_ttm', 'net_profit_margin_ttm','return_on_asset_ttm','return_on_equity_ttm','total_asset_turnover_ttm']
TYPE = 'financial_quality'
quality_factor_dict = {TYPE:QUALITY_FACTORS}

df_factor = add_factors(df_backtest, quality_factor_dict)
df_factor_after_standardize = standardize_factors(df_factor, QUALITY_FACTORS)

# %%
quality_t_tester = TTester()
for factor in QUALITY_FACTORS:
    quality_t_tester.run(df_factor_after_standardize, factor)
    quality_t_tester.get_summary()

#TODO: cannot produce dataframe in for loop？

# %%
quality_IC_tester = ICTester()
for factor in QUALITY_FACTORS:
    quality_IC_tester.run(df_factor_after_standardize, factor)
    quality_IC_tester.get_summary()
    quality_IC_tester.get_graph()

# %%
# one-stop call for a single factor
factor_name = QUALITY_FACTORS[1]
factor_tester = SingleFactorTester(df=df_factor_after_standardize)
factor_tester.t_value_test(factor_name)
factor_tester.IC_test(factor_name)

# %%
df_backtest = df_factor_after_standardize.drop(columns=QUALITY_FACTORS)

# %% [markdown]
# ### Cashflow Factors

# %%
CASHFLOW_FACTORS = ['cash_flow_per_share_ttm','cash_flow_ratio_ttm','fcfe_ttm','ocf_to_debt_ttm','operating_cash_flow_per_share_ttm']
TYPE = 'cashflow'
quality_factor_dict = {TYPE:CASHFLOW_FACTORS}

df_factor = add_factors(df_backtest, quality_factor_dict)
df_factor_after_standardize = standardize_factors(df_factor, CASHFLOW_FACTORS)

# TODO: fix assertion error
# TODO: for function add_factors, maybe change parameters from a factor class dict to a single factor string?

# %%
factor_tester = SingleFactorTester(df=df_factor_after_standardize)
factor_tester.t_value_test(factor_name)
factor_tester.IC_test(factor_name)
