# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3.9.7 ('multifactor')
#     language: python
#     name: python3
# ---

# %%
import os 
os.chdir('../')

import src.dataloader as dl
from src.constants import *

# %%
# Fill in the factor to be downloaded
factor = ['pe_ratio_ttm','pcf_ratio_ttm', 'pcf_ratio_total_ttm']

dl.rq_initialize()
dl.download_factor_data(dl.stock_names, factor, START_DATE, END_DATE)
