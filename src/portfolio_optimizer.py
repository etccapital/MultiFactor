from datetime import datetime, timedelta
import src.dataloader as dl
import pandas as pd
import rqdatac as rq
import scipy
import statsmodels as sm
import numpy as np
import seaborn as sns
import pathos
from tqdm import tqdm
import multiprocessing
import pickle
import matplotlib.pyplot as plt
from src import preprocess
from src.utils import *
from collections import Iterable
import src.factor_combinator as comb
import scipy.sparse as sp
import cvxpy as cp
from src.constants import *

class PortfolioOptimizer:
    """
    A class created specifically for the portfolio optimization process
    For a complete math derivation process, see Huatai MultiFactor Report1 华泰多因子系列1
    """
    def __init__(self, df_backtest: pd.DataFrame, style_factor_dict: dict, hist_periods=12, gamma=1.):
        """       
        Args:
            df_backtest (pd.DataFrame): a pandas dataframe used for backtesting. It has multi-index (date, stock)
            style_factors_dict (dict): a dictionary mapping factor types to factor lists
                e.g. {'value': ['pe_ratio_ttm', 'pb_ratio_ttm', ],
                        }
                in order for factor data to be correctly read in,
                pe_ratio_ttm.h5 and pb_ratio_ttm.h5 should exist under ./Data/factor/value/
            hist_periods (int, optional): number of months of historica data used for forecasting. Defaults to 12.
        """
        self.df_backtest = df_backtest
        self.hist_periods = hist_periods
        self.style_factor_dict = style_factor_dict
        # 'style_factors' is a flatten version of 'style_factor_dict' containing no factor types, only factor names
        self.style_factors = sum(style_factor_dict.values(), []) 
        self.country_factor = 'country'
        self.gamma = gamma

    def run(self, ):
        """
        The main function in this class
        1. select factors and set up the backtesting framework 
        2. calculate the historical factor returns
        3. forecast next period's factor return and covariance matrix, forecast the return and risk of the overall portfolio
        4. set up objective and constraints to solve for the optimal stock portfolio
        5. see the backtesting results
        """
        self.preprocess() #step 1
        self.get_regression_results() #step 2
        self.predict() #step 3
        self.solve_opt_weights() #step 4
        self.plot_return() #step 5

    @timer
    def preprocess(self, ):
        """ 
            Step 1: 
            Preprocess the dataframe before starting any other steps
        Returns:
            The updated backtesting dataframe
        """
        self.df_backtest = preprocess.add_factors(self.df_backtest, self.style_factor_dict)
        self.df_backtest = preprocess.standardize_factors(self.df_backtest, self.style_factors)
        self.df_backtest[self.country_factor] = 1

        # Turn the industry column into one-hot vectors
        df_industry_dummy = pd.get_dummies(self.df_backtest[PRIMARY_INDUSTRY_COL])
        # Set all the industry factors
        self.industry_factors = list(df_industry_dummy.columns)
        self.df_backtest.loc[:, self.industry_factors] = df_industry_dummy.values  
        # Set all the factors
        self.all_factors = [self.country_factor] + self.industry_factors + self.style_factors
        return self.df_backtest

    @timer
    def get_regression_results(self, ):
        """
            Step 2: 
            Obtain the regression results on each rebalancing date, which include historical factor returns and historical idiosyncratic returns
            Store them in pandas dataframes
        Returns:
            A list of statsmodels.wls result objects 
        """

        # Fit a weighted least square regression model on each rebalancing date
        # Regress next period's return with the factor exposures on the current rebalancing date
        # The coefficients are the factor returns in the next period
        def get_wls_result(df_sub):
            import statsmodels.formula.api as smf
            wls_result = smf.wls(formula = f"next_period_return ~ 0 + {self.country_factor} + {' + '.join(self.industry_factors)} + {' + '.join(self.style_factors)}", 
                            data=df_sub, weights = df_sub['market_value'] ** 0.5, missing='drop').fit()
            return wls_result
        # multiprocessing or multithreading is not any faster than for loop here
        self.wls_results = [get_wls_result(df_sub) for _, df_sub in self.df_backtest.groupby(level=0)] 
        date_wls_pairs = zip(self.df_backtest.index.get_level_values(0).unique(), self.wls_results)
        # obtain the historical factor returns
        self.df_hist_factor_return = pd.concat({ date: wls_result.params for date, wls_result in date_wls_pairs}).unstack(level=1)
        self.df_hist_factor_return.index.name = 'date'
        # obtain the idiosyncratic returns
        self.df_hist_idio_return = pd.concat([wls_result.resid for wls_result in self.wls_results])
        return self.wls_results
    
    @timer
    def predict(self, ):
        """
            Step 3:
            Forecast next period's factor return and covariance matrix
        Returns:
            None
        """
        self.df_pred_factor_return = self.predict_factor_return()
        self.df_pred_factor_cov = self.predict_factor_cov()
        #All of the first 12 periods, and no other periods, should have nan values
        return_nan_count_by_date = (self.df_pred_factor_return.notnull().sum(axis=1) == 0).sum()
        assert( return_nan_count_by_date == self.hist_periods)
        cov_nan_count_by_date = (self.df_pred_factor_cov.notnull().groupby(level=0).sum().sum(axis=1) == 0).sum()
        assert( cov_nan_count_by_date == self.hist_periods)

        self.df_pred_idio_return = self.predict_idio_return()
        self.df_pred_stock_returns = self.predict_stock_return()
        
        
    @timer
    def solve_opt_weights(self, ):
        """
            Step 4: Maximize risk-adjusted return to solve for the optimal stock portfolio
        Returns:
            A list of lists: The optimal weights on each rebalancing date
        """
        gamma = cp.Parameter(nonneg=True)  
        Lmax = cp.Parameter()
        def solve_optimal_weight(data, solver=cp.ECOS, abs_tol=1e-8):
            """
            Solve for the optimal weights on a SINGLE rebalancing date.
            """
            X_t, F_t, Delta, r = data
            """
            Let V be the N x N predicted stock return covariance matrix over the next period
            V is predicted as follows: V = X * F * X.transpose() + Delta, where
            X is the N x K factor exposure matrix on the current rebalancing date
            F is the K x K predicted factor covariance matrix over the next period
            Delta is the N x N predicted idiosyncratic return matrix over the next period

            Objective:
            Maximize R - gamma * var, where
            R is the 1 x 1 predicted portfolio return over the next period
            gamma is the 1 x 1 risk penalty coefficient. larger gamma will make the model more inclined to return and less inclined to risk 
            var is the 1 x 1 predicted portfolio variance over the next period 

            Mathematically, we have:
            R = w.transpose * r
            var = w.transpose * V * w
            V is the N x N predicted stock covariance matrix
            w is the N x 1 portfolio weight vector we wish to optimize

            Constraints:
            1) No short-selling: all weights should be non-negative
            2) Stock diversification: Investment into any stock should be less than 1%
            TODO: Add more constraints as outlined in Huatai's report
            """
            gamma.value = self.gamma
            N = X_t.shape[0]
            w = cp.Variable(N)
            ret = r.T @ w
            # Multiplying w and X first gives O(nk^2) time complexity, as opposed to O(n^3) if we calculate V first
            # This saves tons of time!
            variance = cp.quad_form(w.T @ X_t, F_t) + cp.sum_squares(np.sqrt(Delta) @ w) #check this out, see if it generalizes to non-diagonal matrices
 
            problem = cp.Problem(cp.Maximize(ret - gamma * variance), 
                        [cp.sum(w) == 1,
                        0 <= w,
                        w <= 0.01, 
                        ]
                            )
            """
            cvxpy will check that the optimization problem is convex before solving it
            If the optimization problem is not convex, one source of error can be that V is not symmetric semi-positive definite. 
            You can verify this by checking if F and Delta are symmetric semi-positive definite or not.
            See footnote below this class about speeding up the convex optimization problem.
            """
            problem.solve(verbose=False, solver=solver)

            # check that the solver converges to a valid solution i.e. the obtained weight vector satisfies the above imposed constraints
            # Use abs_tol to avoid numerical rounding issues
            solved_weight = w.value
            assert(abs(solved_weight.sum() - 1) < abs_tol)
            assert(np.all(solved_weight >= 0 - abs_tol) )
            assert(np.all(solved_weight <= 0.01 + abs_tol))
            return w.value

        def get_data_by_date(date):
            """Given the rebalacing date, return X, F, Delta and r on that SINGLE rebalancing date.
            """
            X_t = self.df_backtest.loc[self.df_backtest.index.get_level_values(0) == date, self.all_factors].values
            F_t = self.df_pred_factor_cov.loc[self.df_pred_factor_cov.index.get_level_values(0) == date, self.all_factors].values
            u = self.df_pred_idio_return[self.df_pred_idio_return.index.get_level_values(0) == date].values
            Delta = scipy.sparse.diags( u ** 2 )
            r = self.df_pred_stock_returns[self.df_pred_stock_returns.index.get_level_values(0) == date].values
            return [X_t, F_t, Delta, r]
        
        # Store all input data by date in a list, then loop through that list and convex-optimize the weight vectors
        self.input_data = [ get_data_by_date(date) for date in REBALANCING_DATES[self.hist_periods: -1] ] 
        valid_date_mask = self.df_backtest.index.get_level_values(0).isin(REBALANCING_DATES[self.hist_periods:-1])
        self.df_backtest.loc[valid_date_mask, 'opt_weight'] = sum( [list(solve_optimal_weight(input)) for input in tqdm(self.input_data)], [])  # this line of code takes 15s ~ 20s on my laptop
        self.opt_weights = self.df_backtest['opt_weight']
        return self.opt_weights

    @timer
    def plot_return(self, ):
        """
        Step 5: Calculate the cumulative return series and visualize the backtesting results
        TODO: Add more functionalities, such as sharpe ratio, win rate, etc. The best way to achieve this would be to create a 'BacktestingResult' class that takes a cumulative return series as argument, and can plot the relevant figures and print out a table.
        """
        # calculate portfolio return in each period
        self.df_backtest['weighted_return'] = self.df_backtest['next_period_return'].values * self.df_backtest['opt_weight'].values

        # calculate the cumulative portfolio return series
        self.df_portfolio_returns = self.df_backtest.groupby(level=0)['weighted_return'].apply(lambda df: df.sum(skipna=False))
        self.df_portfolio_cum_returns = (1 + self.df_portfolio_returns).cumprod()
        # plot it out
        # sns.distplot( df_portfolio_returns)
        self.df_portfolio_cum_returns.plot()

    def predict_factor_return(self, method=None):
        #helper function for self.predict
        if method is not None:
            #TODO: Implement other advanced methods for forecasting factor returns
            pass
        else:
            #Baseline rolling average prediction
            df_pred_factor_return = self.df_hist_factor_return.rolling(self.hist_periods).mean().shift(1)
        return df_pred_factor_return
        
    def predict_factor_cov(self, method=None):
        #helper function for self.predict
        if method is not None:
            #TODO: Implement other advanced methods for forecasting factor covariance
            pass
        else:
            #Naive covariance matrix estimation
            df_pred_factor_cov = self.df_hist_factor_return.rolling(self.hist_periods).cov().groupby(level=1).shift(1)
        return df_pred_factor_cov
    
    def predict_idio_return(self, method=None):
        #helper function for self.predict
        if method is not None:
            #TODO: Implement other advanced methods for forecasting idiosyncratic return
            pass
        else:
            #No prediction at all, based on the assumption that idiosyncratic return represents the unpredictable portion of stock return
            return pd.Series(0, index=self.df_backtest.index)
    
    def predict_stock_return(self, ) -> pd.Series:
        """
        Helper function for self.predict
        Predict stock return under the factor modelling framework. We do this once for ALL rebalancing dates.
            N is the # of stocks on the current rebalancing date
            K is the # of factors on the current rebalancing date
            r is the predicted N x 1 stock return vector over the next period
            We predict r as follows:
            r = X x f + u
            X is a N x K factor exposure matrix on the current rebalancing date
            f is the predicted K x 1 factor return vector over the next period
            u is the predicted N x 1 idiosyncratic return vector over the next period
        Returns:
            pd.Series: the predicted N x 1 stock return vector over the next period
        """
        X = self.df_backtest[self.all_factors]
        f = pd.merge(self.df_pred_factor_return, X[[]], how='right', left_on='date', right_on='date', )
        u = self.df_pred_idio_return
        # u = pd.merge(X[[]], self.df_pred_idio_return.rename('idio_return'), how='left', left_index=True, right_index=True)['idio_return']
        self.df_pred_stock_returns = pd.Series((X.values * f.values).sum(axis=1) + u.values, index=X.index, name='predicted_stock_return')
        return self.df_pred_stock_returns



"""
Detailed notes on portfolio optimization:

In both factor combination and portfolio optimization problems, we have to implement constrained optimization algorithms to solve for an array of weights. The former problem is easy and can be solved with scipy.optimize within 0.2 seconds for all rebalancing dates. This is because the number of factors to be combined are at most 10 ~ 20, and we are finding solution in a very low dimension. However, based on my empirical testing results, the time it takes to complete an optimization problem usually scales quadratically with the input dimension. This is why it takes more than 10 minutes for scipy to solve a portfolio optimization problem on a single rebalancing date using the naive approach! Since we have 100+ rebalancing dates, the time is definitely intractable if we stay with the naive approach. As a result, below are some trials by me to speed up the optimization problem:

1. Formulate as convex optimization problem and use cvxpy instead of scipy.optimize(Success)
Not sure by how much time this speeds up the problem. But it's said on stackoverflow that cvxpy are usually much faster because the package deals specifically with convex problems.

2. Multiply the matrices in different orders (Very successful). 
Following https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/applications/portfolio_optimization.ipynb#scrollTo=zDfbDngvkJAV. This is the most effective way to speed up the problem! This boosts the solver's time from O(n^3) to O(nk^2), where n is # of stocks and k is # of factors. A single optimization problem used to take around 10 minutes but now takes only 3 seconds!

3. Set up the problem at the beginning and change values in each iteration (Failed)
Because the dimension of matrices are changing in each iteration, it seems like we cannot save the setup time by setting up variables at the beginning and changing the variable values during each iteration.

4. Use a more efficient version of BLAS (Ongoing)
As introduced in https://markus-beuckelmann.de/blog/boosting-numpy-blas.html , there are four versions of BLAS & LAPACK, 
default, ATLAS, IntelMKL, OpenBLAS. It seems that the other three versions are at least around x2 times faster than the default version.
According to https://mruss.dev/numpy-mkl/ and https://conda-forge.org/docs/maintainer/knowledge_base.html#blas, I tried to download the IntelMKL version using conda. The download is sucessful everytime, and 
command line installs the correct package each time. But `numpy.show_config()` gives 'NO ATLAS' for BLAS opt and MKL opt everytime, meaning 
I am still using the default BLAS version for numpy. This is so annoying!

5. Choose the best solver - ECOS (Success)
Setting: for loop + no multiprocessing 
default(OSQP): 1m42s 
SCS: 2m26s
ECOS: 1m10s
ECOS is the fastest

6. Use multiprocessing (Can be implemented but not faster). 
Not sure why it takes the same amount of time when I use for loop v.s. multiprocessing. Plus, the timing functions(either timer.timer, timer.process_time or timer.perf_counter) do not record the true time -- the true time is always longer than what is recorded. Considering the memory issues caused by multiprocessing, we'll use for loop onwwards.

7. Set near-zero values to 0 in the matrix(Failed). 
Originally I thought this may be a good idea because 94%+ values in the matrix have absolute values smaller than 0.0001, so we can approximate those values as 0's, thereby taking the computational advantage of sparse matrices. But it turned out that after setting them as 0, the matrix is no longer semi-positive definitie.
"""