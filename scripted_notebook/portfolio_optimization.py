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
#     language: python
#     name: python3
# ---

# %%
#Don't run this twice!
import os 
os.chdir('../')
from src.constants import *
print(os.getcwd())

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


# %% [markdown]
# Huatai MultiFactor Report1 华泰多因子系列1
# 1. select factors and set up the backtesting framework 
# 2. calculate the historical factor returns
# 3. forecast next period's factor return and covariance matrix
# 4. calculate the return and risk of the overall portfolio
# 5. set up objective and constraints to solve for the optimal stock portfolio
# 6. see the backtesting results

# %%
# def check_symmetric(a, rtol=1e-05, atol=1e-08):
#     assert( np.allclose(a, a.T, rtol=rtol, atol=atol) )

# def check_semi_pos_def(x, atol=1e-8):
#     eig_vals = np.linalg.eigvals(x)
#     # All eigenvalues should be non-negative. Added absolute tolerance here to allow for rounding errors.
#     assert(all(eig_vals >= (0 - atol) ))

def check_symmetric_semi_pos_def(matrix, rtol=1e-05, atol=1e-08):
    # The matrix should be the same as its transpose
    assert( np.allclose(matrix, matrix.T, rtol=rtol, atol=atol) )
    # All eigenvalues should be non-negative. Added absolute tolerance here to allow for rounding errors.
    assert( all(np.linalg.eigvals(matrix) >= (0 - atol) ) )


# %% [markdown]
# #### 1. select factors and set up the backtesting framework 

# %%
STYLE_FACTOR_DICT = {'value': ['pe_ratio_ttm', 'pb_ratio_ttm', 'peg_ratio_ttm', 'book_to_market_ratio_ttm', 'pcf_ratio_ttm', 'ps_ratio_ttm']} 
STYLE_FACTORS = sum( list( STYLE_FACTOR_DICT.values() ), [])

# %%
df_basic_info = dl.load_basic_info()
# filter = preprocess.TimeAndStockFilter(df_basic_info)
# df_backtest = filter.run()

# %%
filter = preprocess.TimeAndStockFilter(df_basic_info)
df_backtest = filter.run()

# %%
df_backtest

# %%
df_backtest = preprocess.add_factors(df_backtest, STYLE_FACTOR_DICT)

# %% [markdown]
# #### 2. calculate the historical factor returns

# %%
## regress next period's return with the factor exposures on the current rebalancing date
## the coefficients are the factor returns in the next period
## store it in a pandas dataframe

# %%
df_backtest = preprocess.standardize_factors(df_backtest, STYLE_FACTORS)

# %%
PRIMARY_INDUSTRY_COL = '一级行业'
INDUSTRY_FACTORS = list(df_backtest[PRIMARY_INDUSTRY_COL].dropna().unique())
COUNTRY_FACTOR = 'country'
ALL_FACTORS = [COUNTRY_FACTOR] + INDUSTRY_FACTORS + STYLE_FACTORS
# industry_rename_mapping = {f"C({PRIMARY_INDUSTRY_COL})[{industry}]": industry for industry in INDUSTRY_FACTORS}

# %%
df_backtest[COUNTRY_FACTOR] = 1

# %%
df_industry_dummy = pd.get_dummies(df_backtest[PRIMARY_INDUSTRY_COL])
df_backtest.loc[:, df_industry_dummy.columns] = df_industry_dummy.values


# %%
def get_factor_return(df_sub):
    import statsmodels.formula.api as smf
    # print(df_sub.index.get_level_values(0)[0])
    df_sub = df_sub.dropna(axis=0)
    wls_result = smf.wls(formula = f"next_period_return ~ 0 + {COUNTRY_FACTOR} + {' + '.join(INDUSTRY_FACTORS)} + {' + '.join(STYLE_FACTORS)}", 
                    data=df_sub, weights = df_sub['market_value'] ** 0.5, missing='drop').fit()
    return wls_result.params

    #code below is for applyParallel
    # df_params = wls_result.params.to_frame()
    # df_params['date'] = df_sub.index.get_level_values(0)[0]
    # df_params = df_params.set_index(['date'], append=True)
    # return df_params

df_factor_return = df_backtest.groupby(level=0).apply(get_factor_return)
# df_factor_returns = applyParallel(df_backtest.groupby(level=0), get_factor_return)

# %%
df_factor_return


# %%
def get_idio_return(df_sub):
    import statsmodels.formula.api as smf
    # print(df_sub.index.get_level_values(0)[0])
    # df_sub = df_sub.dropna(axis=0)
    wls_result = smf.wls(formula = f"next_period_return ~ 0 + {COUNTRY_FACTOR} + {' + '.join(INDUSTRY_FACTORS)} + {' + '.join(STYLE_FACTORS)}", 
                    data=df_sub, weights = df_sub['market_value'] ** 0.5, missing='drop').fit()
    return wls_result.resid
    # return pd.merge(wls_result.resid.rename('resid'), df_sub[[]], how='right', left_on='stock', right_on='stock')
df_idio_return = df_backtest.groupby(level=0).apply(get_idio_return).reset_index(level=1, drop=True)
# df_idio_return = df_idio_return.merge(df_backtest[[]], how='right', left_index=True, )

# %% [markdown]
# 3. forecast next period's factor return, factor covariance matrix, idiosyncratic return, and idiosyncratic covariance matrix

# %%
K = len(STYLE_FACTORS) #number of factors
N = df_backtest.groupby(level=0)['close'].count() #number of stocks, changes over time

# %%
hist_periods = 12

# %%
df_forecasted_factor_return = df_factor_return.rolling(hist_periods).mean()

# %%
df_forecasted_factor_cov = df_factor_return.rolling(hist_periods).cov()

# %%
df_forecasted_factor_cov

# %%
df_cov = df_forecasted_factor_cov[df_forecasted_factor_cov.index.get_level_values(0) == '2020-11-30']

# %%
df_cov.shape
check_symmetric_semi_pos_def(df_cov)

# %%
np.linalg.eigvals(df_cov)

# %%
df_forecasted_idio_return = 0

# %% [markdown]
# 4. calculate the predicted return and risk of the N stocks

# %%
#N is the # of stocks on the current rebalancing date
#K is the # of factors on the current rebalancing date
#r = X x f + u
#X is a N x K factor exposure matrix on the current rebalancing date
#f is the predicted K x 1 factor return vector over the next period
#u is the predicted N x 1 idiosyncratic return vector over the next period
#r is the predicted N x 1 stock return vector over the next period

# %%
sns.distplot( df_backtest['next_period_return'].values)

# %%
X = df_backtest[ALL_FACTORS]
X

# %%
df_factor_return

# %%
f = pd.merge(df_factor_return, X[[]], how='right', left_on='date', right_on='date', )
f

# %%
df_idio_return.rename('idio_return').to_frame()

# %%
u = pd.merge(X[[]], df_idio_return.rename('idio_return'), how='left', left_index=True, right_index=True)['idio_return']
u

# %%
(X.values * f.values).sum(axis=1)

# %%
df_pred_stock_returns = pd.Series((X.values * f.values).sum(axis=1) + u.values, index=X.index, name='predicted_stock_return')

# %%
df_pred_stock_returns

# %%
print(df_pred_stock_returns.shape)

# %% [markdown]
# #### Risk forecasting

# %%
# V = X * F * X.transpose() + Delta
# X is the N x K factor exposure matrix on the current rebalancing date
# F is the K x K predicted factor covariance matrix over the next period
# Delta is the N x N predicted idiosyncratic return matrix over the next period
# V is the N x N predicted stock return covariance matrix over the next period

#the most straightforward way is to use a for loop

# %%
#for loop takes ~40 seconds
#use it for now, update later
V_results = {}
for date in tqdm(REBALANCING_DATES[hist_periods:-1]):
    # date = '2011-01-31'
    print(date)
    X_t = X.loc[X.index.get_level_values(0) == date]
    F_t = df_forecasted_factor_cov.loc[df_forecasted_factor_cov.index.get_level_values(0) == date, ALL_FACTORS]
    check_symmetric_semi_pos_def(F_t)

    u = df_idio_return[df_idio_return.index.get_level_values(0) == date]
    Delta = np.diag(((u.values)**2).flatten())
    check_symmetric_semi_pos_def(Delta)

    #It can be proved that, as long as F_t and Delta are symmetric positive definite matrices, V_t should also be.
    #As a result, we can avoid checking if V_t is a semi-positive matrix, saving a lot of time.
    V_t = X_t.values @ F_t.values @ X_t.transpose().values + Delta
    # V_results[date] = V_t

# %% [markdown]
# 5. set up objective and constraints to solve for the optimal stock portfolio

# %%
REBALANCING_DATES[:-1]

# %%
date = '2020-11-30'
X_t = X.loc[X.index.get_level_values(0) == date]
F_t = df_forecasted_factor_cov.loc[df_forecasted_factor_cov.index.get_level_values(0) == date, ALL_FACTORS]
check_symmetric_semi_pos_def(F_t)

u = df_idio_return[df_idio_return.index.get_level_values(0) == date]
Delta = np.diag(((u.values)**2).flatten())
check_symmetric_semi_pos_def(Delta)

#It can be proved that, as long as F_t and Delta are symmetric positive definite matrices, V_t should also be.
#As a result, we can avoid checking if V_t is a semi-positive matrix, saving a lot of time.
V_t = X_t.values @ F_t.values @ X_t.transpose().values + Delta
N = X_t.shape[0]
r = df_pred_stock_returns[df_pred_stock_returns.index.get_level_values(0) == date]

# %%
#try scipy.optimize first -- too slow, doesn't work

# %%
uniform_weights = [1/N] * N


# %%
def check_valid_portfolio(portfolio_weights):
    #portfolio weights must add up to 1
    assert(abs(sum(portfolio_weights) - 1) < 1e-8)
    #we do not allow short-selling in the portfolio, therefore portfolio weights must be non-negative
    assert((np.array(portfolio_weights) >= 0).all())


# %%
r


# %%
def get_portfolio_return(portfolio_weights):
    return r.values @ portfolio_weights

def get_portfolio_risk(portfolio_weights):
    # check_valid_portfolio(portfolio_weights)
    return (portfolio_weights @ V_t @ portfolio_weights) ** 0.5

def get_objective(portfolio_weights, risk_aversion_coefficient):
    return get_portfolio_return(portfolio_weights) - risk_aversion_coefficient * get_portfolio_risk(portfolio_weights)
    
print( get_portfolio_return(uniform_weights), get_portfolio_risk(uniform_weights), get_objective(uniform_weights, 1) )


# %% [markdown]
# In contrast to around 10 factors in the factor combination process, in the portfolio optimization process there are 1500~4000
# stocks per rebalancing date. That means the dimensionality of the optimization problem is scaled up by an order of magnitude
# of 2.
# As a result, the original method(scipy.optimize.minimize with method = 'SLSQP') takes very long time -- testing it on my macbook pro 
# with intel i9 processor, the problem on a single rebalancing date cannot be solved within 10 minutes. Plus, there are in total
# 120 rebalancing dates.
# We need to find some ways to speed up the optimization.
# Here's what I've done:
# 1) We first need to guarantee that the problem indeed has a solution and the solver can converge to it. Plus, my intuition
#    tells me that when the optimization problem is convex, there should be a much faster solution than the general solver. 
#    So I first checked that all the covariance matrix(no matter predicted ones or empirical ones) were symmetric semi-positive definite -- theoretically they should be!
#    In this examination process I fixed several issues.
# 2) Then I googled some alternative solutions to scipy.optimize. In the following article:
#    https://stackoverflow.com/questions/43648073/what-is-the-fastest-way-to-minimize-a-function-in-python
#    it is said that SLSQP in scipy.optimize is designed for general purpose optimization problems. When the matrix is symmetric semi-positive definite:
#    Using cvxpy + eco/scs would be much faster. An example with a 1000 * 1000 matrix is provided.
# 3) So I pasted the code to this jupyter notebook and wanted to first make sure that the same problem takes <1min to be solved.
#    But cvxpy keeps saying that the problem has no solution, and it takes around 1min10sec for it to do 10000 iterations.
#    I'll try to solve the no-solution problem by increasing max # of iterations. I'll also try using multi-threading to speed up the solver.
# Current progress: Trying to install BLAS and use the best configuration for numpy to speed up cvxpy.
# https://stackoverflow.com/questions/63117123/cvxpy-quadratic-programming-arpacknoconvergence-error
#
# New Idea: 
# --99% of the elements in stock covariance matrix is between -0.005 and 0.005; see if we can treat them as 0 and approximate the solution. Maybe convex optimizers are faster on sparse matrices.

# %%
# from functools import partial
# opt_result = scipy.optimize.minimize(
#                 partial(get_objective, risk_aversion_coefficient=0.25),
#                 uniform_weights,
#                 bounds=[(0, 0.01) for i in range(N)],
#                 constraints=({"type": "eq", "fun": lambda weight: np.sum(weight) - 1})
#             )

# %%
def check_symmetric_semi_pos_def(matrix, rtol=1e-05, atol=1e-08):
    # The matrix should be the same as its transpose
    assert( np.allclose(matrix, matrix.T, rtol=rtol, atol=atol) )
    # All eigenvalues should be non-negative. Added absolute tolerance here to allow for rounding errors.
    assert( all(np.linalg.eigvals(matrix) >= (0 - atol) ) )


# %% [markdown]
# Optimization form 1: Risk Minimization

# %%
for tol in [0.001, 0.003, 0.005, 0.01]:
    print( (np.abs(V_t) <= tol).sum().sum() / V_t.shape[0] ** 2)

# %%
import time
import numpy as np
from cvxpy import * # Convex-Opt

""" Create some random semi-pos-def matrix """

matrix = cvxpy.atoms.affine.wraps.psd_wrap(V_t)
N = V_t.shape[0]

""" CVXPY-based Convex-Opt """
print('\ncvxpy\n')
risk_coefficient = 0.25
x = Variable(N)
constraints = [x >= 0, x <= 0.01, sum(x) == 1]
objective = Minimize(quad_form(x, matrix))
problem = Problem(objective, constraints)
time_start = time.perf_counter()
problem.solve(solver=SCS, use_indirect=True, verbose=True)  # or: solver=ECOS
time_end = time.perf_counter()
print(problem.value)
print('cvxpy (modelling) + ecos/scs (solving) used (secs): ', time_end - time_start)

# %%
# import time
# import numpy as np
# from cvxpy import * # Convex-Opt

# """ Create some random pos-def matrix """
# N = 4000
# matrix_ = np.random.normal(size=(N,N))
# matrix = np.dot(matrix_, matrix_.T)
# matrix = cvxpy.atoms.affine.wraps.psd_wrap(matrix)

# """ CVXPY-based Convex-Opt """
# print('\ncvxpy\n')
# x = Variable(N)
# constraints = [x >= 0, x <= 0.01, sum(x) == 1]
# objective = Minimize(quad_form(x, matrix))
# problem = Problem(objective, constraints)
# time_start = time.perf_counter()
# problem.solve(solver=SCS, use_indirect=True, verbose=True)  # or: solver=ECOS
# time_end = time.perf_counter()
# print(problem.value)
# print('cvxpy (modelling) + ecos/scs (solving) used (secs): ', time_end - time_start)


# %%
opt_result

# %%

# %%

# %%
