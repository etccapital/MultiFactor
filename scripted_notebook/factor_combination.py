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
#     display_name: 'Python 3.7.10 64-bit (''multifactor'': conda)'
#     name: python3
# ---

# %% [markdown]
# #### Set up
# Run this section before anything else


# %%
#Don't run this twice!
import os 
os.chdir('../')
from src.constants import *

# %%
from datetime import datetime, timedelta
import src.dataloader as dl
import pandas as pd
import rqdatac as rq
import scipy
import statsmodels as sm
import numpy as np
import seaborn as sns
import pathos
from tqdm.notebook import tqdm
import multiprocessing
import pickle
import matplotlib.pyplot as plt
from src import preprocess
from src.utils import *
from collections import Iterable
import src.factor_combinator as comb

# %%
df_basic_info = dl.load_basic_info()
filter = preprocess.TimeAndStockFilter(df_basic_info)
df_backtest = filter.run()

# %% [markdown]
# ## Factor Combination

# %%
df_backtest = df_backtest.loc[:, df_backtest.columns.isin(NECESSARY_COLS)]

# %%
value_factors_uniform_combinator = comb.FactorCombinatorUniform(
                          factor_type='value', 
                          factors=['ev_ttm.h5', 'pe_ratio_ttm', 'pb_ratio_ttm', 'peg_ratio_ttm', 'book_to_market_ratio_ttm', 'pcf_ratio_ttm', 'ps_ratio_ttm'], 
                          df_backtest=df_backtest, )

# %%
value_factors_uniform_combinator.run()

# %%
value_factors_ic_combinator = comb.FactorCombinatorByIC(factor_type='value', 
                          factors=['pe_ratio_ttm', 'pb_ratio_ttm', 'peg_ratio_ttm', 'book_to_market_ratio_ttm', 'pcf_ratio_ttm', 'ps_ratio_ttm'], 
                          df_backtest=df_backtest, 
                          hist_periods=12,
                          standardize_factors=True)

# %%
value_factors_ic_combinator.run()

# %%
value_factors_ic_combinator.df_opt_factor_weights

# %%
value_factors_ic_combinator.df_backtest['combined_factor']

# %%
