import os 
from datetime import datetime, timedelta
import src.dataloader as dl
import pandas as pd
import rqdatac as rq
from src.constants import *
from src.utils import *
from src.preprocess import *

# import scipyxf
import statsmodels as sm
import numpy as np
import scipy
import seaborn as sns
from tqdm.notebook import tqdm
import multiprocessing
import pickle
import matplotlib.pyplot as plt
from collections import Iterable

class FactorCombinator:
    """A superclass for all factor combination methods
    """
    def __init__(self, factors: Iterable, factor_type:str,  df_backtest:pd.DataFrame, standardize_factors=False,):
        """_summary_

        Args:
            factors (Iterable): an iterable of strings
            factor_type (str): the type of factor in 'factors' e.g. value
            standardize_factors (bool, optional): whether to standardize the factors before calculting the weights and combining them. Defaults to False.
            df_backtest (pd.DataFrame, optional): (date, stock) multi-index dataframe that specifies all the basic information of each stock on each rebalancing date. 
                                            Must contain at least these columns: 'market_value', 'pri_indus_code', 'next_period_return'
                                            Defaults to df_backtest.
        """
        self.factors = factors
        self.factor_type = factor_type
        self.weight_cols = [factor + '_weight' for factor in factors]
        self.num_factors = len(factors)
        self.uniform_weights = np.array([1 / self.num_factors] * self.num_factors)
        self.standardize_factors = standardize_factors
        self.df_backtest = df_backtest

    @timer
    def preprocess_dataset(self, ):
        """
        Add factor to the backtesting dataset
        Standardize the factors depending on user's choice
        """
        self.df_backtest = add_factors(self.df_backtest, {self.factor_type: self.factors} )
        if self.standardize_factors:
            self.df_backtest = standardize_factors(self.df_backtest, self.factors)

    def get_factor_weights(self, ):
        pass

    @timer
    def combine_factors(self, df_factor_weights: pd.DataFrame):
        """Combine the factors according to given factor weights

        Args:
            df_factor_weights (pd.DataFrame): This dataframe gives the factor weights
                                              Its index should be a subset of the rebalancing dates in self.df_backtest
                                              It must contain all columns in self.weight_cols
                                              The factor weights in each row should be non-negative and add up to 1.
        """
        #the factor weights must be non-negative
        assert( (df_factor_weights.loc[:, self.weight_cols] >= 0).all().all() )
        #the factor weights must add up to 1
        assert( (df_factor_weights.loc[:, self.weight_cols].sum(axis=1) - 1.).abs() < 1e-6 ).all()
        #merge the weights to the backtesting dataframe
        self.df_backtest = self.df_backtest.merge(df_factor_weights.loc[:, self.weight_cols], how='left', left_on='date', right_index=True)
        #vectorized multiplication
        self.df_backtest['combined_factor'] = (self.df_backtest[self.factors].values * self.df_backtest[self.weight_cols].values).sum(axis=1)
        
    def run(self, ) -> pd.DataFrame:
        """the main function in this class
            1. proprocesses the backtesting dataframe
            2. calculates the factor weights
            3. calculates the combined factor column
            4. returns the updated dataframe

        Returns:
            the updated backtesting dataframe with an extra column named 'combined_factor'
        """
        self.preprocess_dataset()
        df_factor_weights = self.get_factor_weights()
        self.combine_factors(df_factor_weights)
        return self.df_backtest.drop(columns=self.weight_cols)

class FactorCombinatorUniform(FactorCombinator):
    """Combines factors with equal weight.
       Benchmark for all other advanced combination methods.
    """
    def get_factor_weights(self, ):
        dates = self.df_backtest.index.get_level_values(0).unique()
        return pd.DataFrame(1/self.num_factors, index=dates, columns=self.weight_cols)

class FactorCombinatorByIC(FactorCombinator):
    """Combines factor exposures according to the weights that maximizes IC value
       See Huatai MultiFactor Report #10 华泰金工多因子系列研报-10
    """
    def __init__(self, hist_periods:int=12, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.hist_periods = hist_periods

    @timer 
    def get_ic_series(self, ):
        """Sets and returns a dataframe of ic values for each factor. The dataframe uses rebalancing dates as index
           and factor names as columns.
           Computes the IC values with multiprocessing.
        """
        def set_ic_value(df_sub: pd.DataFrame) -> float:
            """IC value is calculated as the rank correlation between the factor residual and next period's return,
               where factor residuals are defined as the residuals of linearly regressing next period's return against market cap and industry factor
               Factor residualization is a form of factor purification; the aim is to remove the factor's linear dependency on market factor and 
               industry factor, exposing the factor's very original state. 
            Args:
                df_sub (pd.DataFrame): a sub dataframe corresponding to a given factor and rebalancing date

            Returns:
                float: IC value
            """
            #The multiprocessing used takes forever, not sure why. I had to use for loop.
            # 2022.02.27 Update by Polo:
            # This was because I used nested processes -- I tried to parallelize both calls to 'get_ic_series'(factor level) and calls inside 'get_ic_series'(date level)
            # Currently, the best solution I know is to flatten 'get_ic_series' into one level 
            # i.e. make a new function 'get_ic_series' that takes both factor and date as inputs,
            # then parallelize calls to this function. 
            #TODO: use the shared_memory functionality developed in python 3.8 for multiple processes
            #      otherwise each subprocess will make copies of original dataframe causing lots of unnecessary extra memory/overhead
            factor = df_sub.columns[-1]
            import statsmodels.formula.api as smf
            factor_resid = smf.wls(formula = f"{factor} ~ 0 + market_value + C(pri_indus_code)", 
                            data=df_sub, weights = df_sub['market_value'] ** 0.5).fit().resid
            ic_val = pd.concat([df_sub['next_period_return'], factor_resid], axis=1).corr(method='spearman').iloc[0, 1]
            return ic_val
            # df_ic_series.loc[date, factor] = ic_val

        def get_df_sub(date, factor):
            return self.df_backtest.loc[date, ['market_value', 'pri_indus_code', 'next_period_return'] + [factor]]
        inputs = [(date, factor) for date in self.df_backtest.index.get_level_values(0).unique() for factor in self.factors]
        with pathos.multiprocessing.Pool(pathos.helpers.cpu_count()) as pool:
            results = pool.map(set_ic_value, [get_df_sub(date, factor) for date, factor in inputs])
        # with pathos.pools._ProcessPool(pathos.helpers.cpu_count()) as pool:
        #     results = pool.starmap(set_ic_value, inputs)
        self.df_ic_series = pd.Series(results, index=pd.MultiIndex.from_tuples(inputs), ).unstack(level=1)
        return self.df_ic_series

    @timer
    def get_factor_weights(self, ) -> pd.DataFrame:
        """
        For each factor:
        1. On each rebalancing date, calculates the mean and covariance matrix of IC values over the past 12 months.
        2. Solves a convex optimization problem to determine which set of factor weights gives the highest expected IC value
           for the combined factor. Here IC values are assumed to be linearly addable and scalable.
        Returns:
            pd.DataFrame: A dataframe giving the optimal factor weights
                            Its index should be a subset of the rebalancing dates in self.df_backtest
                            It must contain the weight columns in self.weight_cols
                            The factor weights in each row should be non-negative and add up to 1.
        """
        self.get_ic_series()
        df_ic_hist_mean = self.df_ic_series.rolling(self.hist_periods, min_periods=1).mean().iloc[self.hist_periods:]
        df_ic_cov_mat_series = self.df_ic_series.rolling(self.hist_periods, min_periods=1).cov().iloc[self.hist_periods * self.num_factors:]
        self.df_opt_factor_weights = pd.DataFrame([], columns=self.weight_cols + ['uniform_IC', 'max_IC'])
        #we cannot use pandas.rolling.apply(func) because rolling.apply is different from groupby.apply -- it cannot take a dataframe as the parameter
        #for loop is pretty fast, so no need to use multiprocessing
        for date in df_ic_hist_mean.index:
            df_ic = df_ic_hist_mean.loc[date, :]
            df_ic_cov_mat = df_ic_cov_mat_series[df_ic_cov_mat_series.index.get_level_values(0) == date]
            # print(f"ICIR with uniform weights: {get_ic_ir(uniform_weights)}")

            def get_ic_ir(factor_weights):
                ic_mean = factor_weights.transpose() @ df_ic.values.flatten()
                ic_var = factor_weights @ df_ic_cov_mat.values @ factor_weights.transpose()
                return ic_mean / (ic_var ** 0.5)

            opt_result = scipy.optimize.minimize(
                            lambda w: -get_ic_ir(w),
                            self.uniform_weights,
                            bounds=[(0, 1)] * self.num_factors,
                            constraints=({"type": "eq", "fun": lambda weight: np.sum(weight) - 1})
                        )

            self.df_opt_factor_weights.loc[date, :] = list(opt_result.x) + [get_ic_ir(self.uniform_weights), get_ic_ir(opt_result.x)]
            # print(f"ICIR with optimal weights: {get_ic_ir(opt_factor_weight)}")
            
        self.df_opt_factor_weights = self.df_opt_factor_weights.astype('float64')

        return self.df_opt_factor_weights[self.weight_cols]