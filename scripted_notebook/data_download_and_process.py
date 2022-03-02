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
import os 
os.chdir('../')

# %%
from datetime import datetime, timedelta
import src.dataloader as dl
from src import preprocess
import pandas as pd
import rqdatac as rq
from src.constants import *
import scipy
import statsmodels as sm
import numpy as np
import seaborn as sns
import pathos
from tqdm.notebook import tqdm
import multiprocessing
import pickle
import matplotlib.pyplot as plt
from src.utils import *
from concurrent.futures import ThreadPoolExecutor

# %%
dl.load_industry_mapping()

# %% [markdown]
# ### Rewrite the data getter code into class form

# %%
df_basic_info = dl.load_basic_info()

# %% [markdown]
# #### Load Index Data
# In hiearachical backtesting we need weights of index(e.g. CSI300) data to make the portfolio to stay industry-neutral with the index.
#
# Currently index data is assumed to be uniformly weighted among all stocks

# %%
df_index = dl.load_index_data()

# %% [markdown]
#     1) Filter out data before START_DATE and after END_DATE(backtesting period) from the raw stock data. 
#
#     - 剔除不在回测区间内的股票信息
#
#     2) Filter listed stocks
#
#     - 选出回测区间内每只股票上市的时间。这一步是为了步骤3，因为在每个选股日筛选ST或者停牌股的前提是股票在该选股日已上市。
#
#     3) Filter out ST stocks, suspended stocks and stocks that are listed only within one year
#     - 剔除ST，停牌和次新股（上市未满一年的股票）

# %%
# df_pri_indus_names = pd.read_excel(os.path.join(DATAPATH, 'raw_data', 'industry_code_to_names.xlsx'), 'Primary Industries')
# df_secon_indus_names = pd.read_excel(os.path.join(DATAPATH, 'raw_data', 'industry_code_to_names.xlsx'), 'Secondary Industries')
# df_indus_mapping = pd.read_hdf("./Data/raw_data/industry_mapping.h5", key='industry_mapping').reset_index().rename(columns={'index': 'stock'})
# df_indus_mapping.merge(df_pri_indus_names, how='left', left_on='pri_indus_code', right_on='pri_indus_code' )
# df_indus_mapping = df_indus_mapping.merge(df_secon_indus_names, how='left', left_on='secon_indus_code', right_on='secon_indus_code' )
# df_indus_mapping = df_indus_mapping.set_index('stock')

# %%
filter = preprocess.TimeAndStockFilter(df_basic_info)
df_backtest = filter.run()

# %%
df_backtest

# %%
# value_factors = ['pe_ratio_ttm', 'pb_ratio_ttm', 'pcf_ratio_ttm', 'peg_ratio_ttm', 'ev_ttm']
# all_factors = {'value': value_factors,
#               }
# df_backtest = preprocess.add_factors(df_backtest, all_factors)

# %%
# df_standardized = preprocess.standardize_factors(df_backtest, value_factors,)
# df_standardized

# %%
value_factors = ['pe_ratio_ttm', 'pb_ratio_ttm', 'pcf_ratio_ttm', 'peg_ratio_ttm', 'ev_ttm']
all_factors = {'value': value_factors,
              }

# %%
df_backtest = preprocess.add_factors(df_backtest, all_factors)

# %%
df_backtest

# %%

# %% [markdown]
# # Single-Factor Backtesting

# %% [markdown]
# 同一类风格因子选多个(5-10)集中测试，下面是常见的几种测试方法，对于每一个方法相应的不同评价指标，我们最后用折线图/柱状图和一张完整的表格/dataframe来展示，详见华泰研报单因子测试中的任意一篇
#
# - 回归法
# - IC值
# - 分层回测（既要与基准组合比较，也要比较超额收益的时间序列）

# %%
SINGLE_FACTOR = 'PE_TTM'

# %%
df_backtest


# %% [markdown]
# # T-Value Analysis
# 回归法

# %%
def wls_tval_coef(df):
    #obtain the t-value in WLS of the tested factor
    # 函数内需要用包要额外在这里加上
    import statsmodels.formula.api as smf
    import pandas as pd
    SINGLE_FACTOR = 'PE_TTM'

    # Weighted Least Square(WLS) uses the square root of market cap of each stock
    # 使用加权最小二乘回归，并以个股流通市值的平方根作为权重
    # other than the factor of interest, we also regress on the industry for neutralization
    # 同时对要测试的因子和行业因子做回归（个股属于该行业为1，否则为0），消除因子收益的行业间差异
    wls_result = smf.wls(formula = f"next_period_return ~ pri_indus_code + {SINGLE_FACTOR}", 
                    data=df, weights = df['market_value'] ** 0.5).fit()
    result_tval_coef = pd.Series( {'t_value': wls_result.tvalues.values[0], 'coef': wls_result.params.values[0], 
                         } )
    # result_resid = pd.Series( {'resid': wls_result.resid.values} )
    return result_tval_coef.to_frame().transpose()


# %%
#get the t-value for all periods
wls_results_tval_coef = applyParallel(df_backtest.groupby(level=0), wls_tval_coef)
wls_results_tval_coef.index = df_backtest.index.get_level_values(level=0).unique()

# %%
# get a summary result from the t-value series
# 回归法的因子评价指标

# t值序列绝对值平均值
tval_series_mean = wls_results_tval_coef['t_value'].abs().mean()
# t 值序列绝对值大于 2 的占比
large_tval_prop = (wls_results_tval_coef['t_value'].abs() > 2).sum() / wls_results_tval_coef.shape[0]
# t 值序列均值的绝对值除以 t 值序列的标准差
standardized_tval = wls_results_tval_coef['t_value'].mean() / wls_results_tval_coef['t_value'].std()
# 因子收益率序列平均值
coef_series_mean = wls_results_tval_coef['coef'].mean()
# 因子收益率均值零假设检验的 t 值
coef_series_t_val = scipy.stats.ttest_1samp(wls_results_tval_coef['coef'], 0).statistic

# %%
print('t值序列绝对值平均值：', '{:0.4f}'.format(tval_series_mean))
print('t值序列绝对值大于2的占比：', '{percent:.2%}'.format(percent = large_tval_prop))
print('t 值序列均值的绝对值除以 t 值序列的标准差：', '{:0.4f}'.format(standardized_tval))
print('因子收益率均值：', '{percent:.4%}'.format(percent=coef_series_mean))
print('因子收益率均值零假设检验的 t 值：', '{:0.4f}'.format(coef_series_t_val))


# %% [markdown]
# ## Information Coefficient Analysis

# %%
#Rank IC is defined by the spearman correlation of the factor residual(after market-value and industry neutralizations)
#with next period's return

# %%
# data preprocess of IC analysis
# 因子值IC值计算之前的预处理
# 因子值在去极值、标准化、去空值处理后，在截面期上用其做因变量对市值因子及行业
# 因子（哑变量）做线性回归，取残差作为因子值的一个替代

def wls_factor_resid(df):
    import statsmodels.formula.api as smf
    wls_result = smf.wls(formula = f"PE_TTM ~ market_value + pri_indus_code", 
                    data=df).fit()
    return wls_result.resid


# %%
factor_resids = applyParallel(df_backtest.groupby(level=0), wls_factor_resid)
factor_resids = factor_resids.rename('PE_TTM_resid')
factor_resids

# %%
df_backtest = df_backtest.merge(factor_resids, how='left', left_index=True, right_index=True)

# %%
df_backtest


# %%
# 下一期所有个股的收益率向量和当期因子的暴露度向量的相关系数
# use Spearman's rank correlation coefficient by default. Another choice is Pearson
def cross_sectional_ic(df):
    return df[['next_period_return', 'PE_TTM_resid']].corr(method='spearman').iloc[0, 1]
ic_series = df_backtest.groupby(level=0).apply(cross_sectional_ic)

# %%
ic_series

# %%
ic_series_mean = ic_series.mean()
ic_series_std = ic_series.std()
ir = ic_series_mean / ic_series_std
ic_series_cum = ic_series.cumsum()
ic_pos_prop = (ic_series > 0).sum() / ic_series.shape[0]

# %%
print('IC 均值:','{:0.4f}'.format(ic_series_mean))
print('IC 标准差:','{:0.4f}'.format(ic_series_std))
print('IR 比率:','{percent:.2%}'.format(percent=ir))
print('IC 值序列大于零的占比:','{percent:.2%}'.format(percent=ic_pos_prop))

# %%
# IC 值累积曲线——随时间变化效果是否稳定
ic_series_cum.plot()

# %%
df_backtest = df_backtest.drop(columns = ['PE_TTM_resid'])

# %% [markdown]
# ## Hiearachical Backtesting
# 分层回测

# %%
NUM_GROUPS = 5
GROUP_NAMES = [f"group{i}_weight" for i in range(1, NUM_GROUPS + 1)]

# %%
from functools import partial
assigned_group = df_backtest.groupby('pri_indus_code')['PE_TTM'].apply(partial(pd.qcut, q=5, labels=range(5)))

# %%
fig = plt.figure(figsize = (10, 5))
num_stocks_per_indus = df_backtest[df_backtest.index.get_level_values(0) == '2011-01-31'].groupby('pri_indus_code')['PE_TTM'].count()

plt.bar(*zip(*num_stocks_per_indus.items()))
 
plt.xlabel("Industry")
plt.ylabel("Number of Companies")
plt.show()

# %%
#Here for simplicity we assume that index weight is a uniform portfolio over all stocks
#TODO: use the real index constituent weights instead
df_backtest['index_weight'] = df_backtest.groupby(level=0).apply(lambda df: pd.Series([1/df.shape[0]] * df.shape[0])).values


# %%
#merge the industry weights of the index onto the backtesting dataframe
def set_index_indus_weight(df):
    index_indus_weight = df.groupby('pri_indus_code')['index_weight'].sum().rename('index_indus_weight')
    df = df.merge(index_indus_weight, how='left', left_on='pri_indus_code', right_index=True)
    return df
#'index_indus_weight' = this stock's industry weight in the benchmark index
if 'index_indus_weight' not in df_backtest.columns:
    df_backtest = applyParallel(df_backtest.groupby(level=0), set_index_indus_weight)

# %%
df_backtest

# %%
import numpy as np
# hiearchical backtesting is pretty hard to implement using purely vectorization/parrallelization, and I have to use for loop at least once.
def get_group_weight_by_industry(num_stocks, num_groups) -> np.array:
    """
    precondition: the stocks need to be sorted by factor exposure
    @num_stocks: the number of stocks in this industry
    @num_groups: the number of portfolio groups to be constructed
    
    returns: an intermediary (num_stocks x num_groups) weight matrix specifying the weight of 
             each stock in each group. Here weights within each group(column sum) adds up to 1.
             This is not the final weight matrix because there are many industries(so that weights within 
             each group should actually be smaller than one) but the returned
             weight matrix represents only one industry. 
    
    if you want to understand the algorithm deeper, print some intermediary outputs
    """
    num_rows = min(num_groups, num_stocks)
    num_cols = max(num_groups, num_stocks)
    weight_mat = np.zeros((num_rows, num_cols))
    remaining = 0
    j = 0
    row_budget = num_cols
    col_budget = num_rows
    for i in range(num_rows):
        # print(f"i = {i}")
        start = col_budget - remaining
        # print(f"start = {start}")
        weight_mat[i, j] = start
        offset = (row_budget - start) // col_budget
        # print(f"offset = {offset}")
        weight_mat[i, j + 1: j + 1 + offset] = col_budget
        remaining = row_budget - offset * col_budget - start
        j = j + 1 + offset
        if j < num_cols:
            weight_mat[i, j] = remaining
        
    weight_mat = weight_mat if num_groups > num_stocks else weight_mat.transpose()
    weight_mat_normalized = weight_mat / weight_mat.sum(axis=0)
    return weight_mat_normalized

def get_weight_df_by_industry(df: pd.DataFrame) -> pd.DataFrame:
    """get the weight dataframe for each industry"""
    #sort by the factor exposure
    df = df.sort_values(by='PE_TTM') 
    stock_names = df.index.get_level_values(1)
    #get weight matrix first
    weight_mat = get_group_weight_by_industry(stock_names.shape[0], NUM_GROUPS)
    df[GROUP_NAMES] = weight_mat
    return df


# %%
def get_group_weight_by_date(df_backtest_sub):
    #get the intermediary weights in each group on each rebalancing date
    df_backtest_sub = df_backtest_sub.groupby('pri_indus_code').apply(get_weight_df_by_industry).droplevel(0).sort_index(level=1)
    """
    we need to make the group portfolio industry-neutral with the index. That is, industry weights should be the same in both
    the group portfolio and the index. 
    """
    #multiply each stock's intermediary weight by its industry weight. since the intermediary weight within each group within each industry adds up to 1(as explained in the previous function),
    #after this operation the final stock weight within each group should add up to 1.
    df_backtest_sub[GROUP_NAMES] = np.multiply(df_backtest_sub[GROUP_NAMES].values, df_backtest_sub['index_indus_weight'].values[:, np.newaxis])
    return df_backtest_sub


# %%
df_backtest = df_backtest.groupby(level=0).apply(get_group_weight_by_date)

# %%
df_backtest

# %%
df_backtest.groupby(level=0)[GROUP_NAMES].sum() #looks good


# %%
def get_group_returns_by_date(df_backtest_sub):
    group_returns = df_backtest_sub[GROUP_NAMES].values.transpose() @ df_backtest_sub['next_period_return'].values
    group_returns = pd.Series(group_returns, index=GROUP_NAMES)
    return group_returns

group_returns_by_date = df_backtest.groupby(level=0).apply(get_group_returns_by_date)
group_returns_by_date

# %%
group_cum_returns= (group_returns_by_date + 1).cumprod(axis=0)
group_cum_returns

# %%
group_cum_returns.plot()

# %% [markdown]
# ## Factor Combination

# %%
df_backtest.columns

# %%
# COMBINE_FACTORS = ['PE_TTM', 'PS_TTM', 'PC_TTM']
COMBINE_FACTORS = ['pb_ratio_ttm', 'pe_ratio_ttm', 'pcf_ratio_ttm']


# %%
def get_ic_series(factor, df_backtest=df_backtest):
    def wls_factor_resid(df):
        import statsmodels.formula.api as smf
        wls_result = smf.wls(formula = f"{factor} ~ 0 + market_value + C(pri_indus_code)", 
                        data=df, weights = df['market_value'] ** 0.5).fit()
        return wls_result.resid
    if f'{factor}_resid' not in df_backtest.columns:
        factor_resids = applyParallel(df_backtest.groupby(level=0), wls_factor_resid)
        factor_resids = factor_resids.rename(f'{factor}_resid')
        df_backtest = df_backtest.merge(factor_resids, how='left', left_index=True, right_index=True)
    def cross_sectional_ic(df):
        return df[['next_period_return', f'{factor}_resid']].corr(method='spearman').iloc[0, 1]
    ic_series = df_backtest.groupby(level=0).apply(cross_sectional_ic)
    return ic_series


# %%
df_backtest

# %%
#The multiprocessing takes forever, not sure why
#had to use for loop for now. Look into this later

# with pathos.multiprocessing.ProcessPool(pathos.helpers.cpu_count()) as pool: 
#     ic_series_results = pool.map( get_ic_series, COMBINE_FACTORS)
ic_series_results = [get_ic_series(factor).rename(factor) for factor in COMBINE_FACTORS] #around 8 seconds
df_ic_series = pd.concat(ic_series_results, axis=1)

# %%
df_ic_series

# %%
hist_periods = 12
df_ic_hist_mean = df_ic_series.rolling(hist_periods, min_periods=1).mean().iloc[hist_periods:]
df_ic_hist_mean

# %%
#leave the computation for later
# A = np.arange(6).reshape(2,3) # 2 x 3, N = 2, T = 3
# S = pd.DataFrame(A).T.cov() #2 x 2
# np.expand_dims(A, 0)

# %%
df_ic_cov_mat_series = df_ic_series.rolling(hist_periods, min_periods=1).cov().iloc[hist_periods:]
df_ic_cov_mat_series

# %% [markdown]
# #### maximize the ICIR values on a single rebalancing date

# %%
df_ic_hist_mean = df_ic_series.rolling(hist_periods, min_periods=hist_periods).mean()

# %%
date = '2011-01-31'

# %%
df_ic = df_ic_series[df_ic_series.index == date]
df_ic

# %%
df_ic_cov_mat = df_ic_cov_mat_series[df_ic_cov_mat_series.index.get_level_values(0) == date]
df_ic_cov_mat


# %%
def get_ic_ir(factor_weights):
    ic_mean = factor_weights.transpose() @ df_ic.values.flatten()
    ic_var = factor_weights @ df_ic_cov_mat.values @ factor_weights.transpose()
    return ic_mean / (ic_var ** 0.5)


# %%
num_factors = len(COMBINE_FACTORS)
uniform_weights = np.array([1 / num_factors] * num_factors)
get_ic_ir(uniform_weights)

# %%
opt_result = scipy.optimize.minimize(
                lambda w: -get_ic_ir(w),
                np.array([1 / num_factors] * num_factors),
                bounds=[(0, 1) for i in range(num_factors)],
                constraints=({"type": "eq", "fun": lambda weight: np.sum(weight) - 1})
            )
opt_factor_weight = opt_result.x

# %%
opt_factor_weight

# %%
get_ic_ir(opt_factor_weight)

# %% [markdown]
# #### optimal factor weight on all rebalancing dates

# %%
df_ic_series

# %%
opt_factor_weights = pd.DataFrame([], columns=COMBINE_FACTORS)
for date in tqdm(df_ic_series.index):
    print(date)
    df_ic = df_ic_series[df_ic_series.index == date]
    df_ic_cov_mat = df_ic_cov_mat_series[df_ic_cov_mat_series.index.get_level_values(0) == date]
    num_factors = len(COMBINE_FACTORS)
    uniform_weights = np.array([1 / num_factors] * num_factors)
    print(f"ICIR with uniform weights: {get_ic_ir(uniform_weights)}")

    opt_result = scipy.optimize.minimize(
                    lambda w: -get_ic_ir(w),
                    np.array([1 / num_factors] * num_factors),
                    bounds=[(0, 1) for i in range(num_factors)],
                    constraints=({"type": "eq", "fun": lambda weight: np.sum(weight) - 1})
                )
    opt_factor_weight = opt_result.x
    opt_factor_weights.loc[date, :] = opt_factor_weight
    print(f"ICIR with optimal weights: {get_ic_ir(opt_factor_weight)}")
    get_ic_ir(opt_factor_weight)

# %%
df_ic_series.rolling(12).apply(lambda df: df.mean() )

# %%
df_ic_series.index[hist_periods:]


# %%
def get_opt_factor_weight_by_date(date): 
    df_ic = df_ic_series[df_ic_series.index == date]
    df_pred_ic = df_ic.rolling(hist_periods).mean()
    print('11111')
    print(df_pred_ic)
    df_pred_ic_cov_mat = df_ic.rolling(hist_periods).cov()
    print('a')
    def get_ic_ir(factor_weights):
        combined_ic_mean = factor_weights.transpose() @ df_pred_ic.values.flatten()
        combined_ic_var = factor_weights @ df_pred_ic_cov_mat.values @ factor_weights.transpose()
        combined_ir = combined_ic_mean / (combined_ic_var ** 0.5)
        return combined_ir
    print('b')
    opt_result = scipy.optimize.minimize(
                lambda w: -get_ic_ir(w),
                np.array([1 / num_factors] * num_factors),
                bounds=[(0, 1) for i in range(num_factors)],
                constraints=({"type": "eq", "fun": lambda weight: np.sum(weight) - 1})
            )
    print('c')
    opt_factor_weight = pd.Series(opt_result.x, index=COMBINE_FACTORS, name=date)
    print('d')
    return opt_result.x


# %%
valid_dates = df_ic_series.index[hist_periods:]
results = [get_opt_factor_weight_by_date(date) for date in valid_dates]

# %%
results #TBD: not sure why the optimization is not working, debug

# %%
#we cannot use pandas.rolling.apply(func) because rolling.apply is different from groupby.apply -- it cannot take a dataframe as the parameter


# num_factors = len(COMBINE_FACTORS)
# def get_opt_factor_weight_by_date(ic_series: pd.Series): 
#     df = df_ic_series.loc[ic_series.index, :]
#     df_pred_ic = df.mean()
#     print('11111')
#     print(df_pred_ic)
#     df_pred_ic_cov_mat = df.cov()
#     print('a')
#     def get_ic_ir(factor_weights):
#         combined_ic_mean = factor_weights.transpose() @ df_pred_ic.values.flatten()
#         combined_ic_var = factor_weights @ df_pred_ic_cov_mat.values @ factor_weights.transpose()
#         combined_ir = combined_ic_mean / (combined_ic_var ** 0.5)
#         return combined_ir
#     print('b')
#     opt_result = scipy.optimize.minimize(
#                 lambda w: -get_ic_ir(w),
#                 np.array([1 / num_factors] * num_factors),
#                 bounds=[(0, 1) for i in range(num_factors)],
#                 constraints=({"type": "eq", "fun": lambda weight: np.sum(weight) - 1})
#             )
#     print('c')
#     date = df.index[0]
#     opt_factor_weight = pd.Series(opt_result.x, index=COMBINE_FACTORS, name=date)
#     print('d')
#     return opt_result.x

# df_ic_series.rolling(12).apply(get_opt_factor_weight_by_date)
