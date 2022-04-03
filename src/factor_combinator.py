from ast import Pass
import os 
from datetime import datetime, timedelta
import pandas as pd
import rqdatac as rq
import src.dataloader as dl
from src.constants import *
from src.utils import *
from src.preprocess import *

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
    """
    A superclass for all factor combination methods
    """
    def __init__(self, factors: Iterable, factor_type:str,  df_backtest:pd.DataFrame, standardize_factors=False,):
        """_summary_

        Args:
            factors (Iterable): an iterable of strings
            factor_type (str): the type of factor in 'factors' e.g. value
            standardize_factors (bool, optional): whether to standardize the factors before calculting the weights and combining them. Defaults to False.
            df_backtest (pd.DataFrame, optional): (date, stock) multi-index dataframe that specifies all the basic information of each stock on each rebalancing date. 
                                            Must contain at least these columns: 'market_value', primary industry, 'next_period_return'
                                            Defaults to df_backtest.
        """
        self.factors = factors
        self.factor_type = factor_type
        self.weight_cols = [factor + '_weight' for factor in factors] # column name of weight dataframe
        self.num_factors = len(factors)
        self.uniform_weights = np.array([1 / self.num_factors] * self.num_factors)
        self.standardize_factors = standardize_factors
        self.df_backtest = df_backtest

    @timer
    def preprocess_dataset(self, ):
        """
        1. Add factor to the backtesting dataset
        2. Standardize the factors depending on user's choice
        """
        self.df_backtest = add_factors(self.df_backtest, {self.factor_type: self.factors} )
        if self.standardize_factors:
            self.df_backtest = standardize_factors(self.df_backtest, self.factors)

    def get_factor_weights(self, ):
        """
        Polymorphism for child classes of different combination methods(uniform, maxIC, maxICIR, etc.)
        """
        pass

    @timer
    def combine_factors(self, df_factor_weights: pd.DataFrame):
        """
        Combine the factors according to given factor weights

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
        """
        the main function in this class with the following steps:
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
    """
    Combines factors with equal/uniform weight.

    Note: Uniform combination is the benchmark for all other more sophisticated combination methods.
    """
    def get_factor_weights(self, ):
        dates = self.df_backtest.index.get_level_values(0).unique()
        return pd.DataFrame(1/self.num_factors, index=dates, columns=self.weight_cols)

class FactorCombinator_Max_IC_or_ICIR(FactorCombinator):
    """
    Combines factor exposures according to the weights that maximizes IC/ICIR value

    For detailed calculation formulas， see Huatai MultiFactor Report #10 华泰金工多因子系列之十：因子合成方法实证分析 
    """
    def __init__(self, hist_periods:int=12, max_what='ICIR', *args, **kwargs):
        super().__init__(*args, **kwargs)
        # choose 12 months as the historical periods
        self.hist_periods = hist_periods
        self.max_what = max_what

    @timer 
    def get_ic_series(self, ):
        """
        Sets and returns a dataframe of ic values for each factor. The dataframe uses rebalancing dates as index
        and factor names as columns.
        Computes the IC values with multiprocessing.
        """
        def set_ic_value(df_sub: pd.DataFrame) -> float:
            """
            IC value is calculated as the rank correlation between the factor residual and next period's return,
            where factor residuals are defined as the residuals of linearly regressing next period's return against market cap and industry factor.

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
            # get factor residuals
            factor_resid = smf.wls(formula = f"{factor} ~ 0 + market_value + C({PRIMARY_INDUSTRY_COL})", 
                            data=df_sub, weights = df_sub['market_value'] ** 0.5).fit().resid
            # get RankIC
            ic_val = pd.concat([df_sub['next_period_return'], factor_resid], axis=1).corr(method='spearman').iloc[0, 1]
            return ic_val

        def get_df_sub(date, factor):
            """
            Return a sub-dataframe with columns of market_value, primary_industry, next_period_return, and the factor.
            """
            return self.df_backtest.loc[date, ['market_value', PRIMARY_INDUSTRY_COL, 'next_period_return'] + [factor]]

        inputs = [(date, factor) for date in self.df_backtest.index.get_level_values(0).unique() for factor in self.factors]
        with pathos.multiprocessing.Pool(pathos.helpers.cpu_count()) as pool:
            results = pool.map(set_ic_value, [get_df_sub(date, factor) for date, factor in inputs])
        self.df_ic_series = pd.Series(results, index=pd.MultiIndex.from_tuples(inputs), ).unstack(level=1)
        return self.df_ic_series

    @timer
    def get_factor_weights(self,) -> pd.DataFrame:
        """
        For each factor:
        1. On each rebalancing date, calculates the mean and covariance matrix of IC values over the past 12 months.
        2. Solves a convex optimization problem to determine which set of factor weights gives the highest expected IC value
           for the combined factor. Here IC values are assumed to be linearly addable and scalable.

        Returns:
            pd.DataFrame: A dataframe giving the optimal factor weights.
                        Its index should be a subset of the rebalancing dates in self.df_backtest
                        It must contain the weight columns in self.weight_cols
                        The factor weights in each row should be non-negative and add up to 1.

        Note:
        we cannot use pandas.rolling.apply(func) because rolling.apply is different from groupby.apply -- it cannot take a dataframe as the parameter
        for loop is pretty fast, so no need to use multiprocessing

        TODO:补充 Ledoit & Wolf(2004)提出的协方差矩阵压缩估计方法
        """
        #get IC dataframe of all factors at all rebalancing dates
        self.get_ic_series()

        #mean IC of all factors within the historical time window
        df_ic_hist_mean = self.df_ic_series.rolling(self.hist_periods, min_periods=1).mean().iloc[self.hist_periods:]
        
        if self.max_what == 'ICIR':

            #covariance matrix of ICs of all factors i.e. Sigma in the paper
            df_cov_mat_series = self.df_ic_series.rolling(self.hist_periods, min_periods=1).cov().iloc[self.hist_periods * self.num_factors:]
            
            #create an empty container for the optimized weights w, uniform IC values and 
            self.df_opt_factor_weights = pd.DataFrame([], columns=self.weight_cols + ['uniform_ICIR', 'max_ICIR'])

        if self.max_what == 'IC':
            #covariance/correlation matrix of factor values
            
            def corr(df):
                return df.corr(method='pearson')

            factor_df = self.df_backtest[self.factors]
            dates = self.df_backtest.index.get_level_values(0).unique().tolist()

            with pathos.multiprocessing.Pool(pathos.helpers.cpu_count()) as pool:
                df_cov_mat_series = pd.concat(pool.map(corr, [factor_df.xs(date,level='date') for date in dates]), keys=dates)
            
            #create an empty container for the optimized weights w, uniform IC values and 
            self.df_opt_factor_weights = pd.DataFrame([], columns=self.weight_cols + ['uniform_IC', 'max_IC'])
        
        for date in df_ic_hist_mean.index:
            df_ic = df_ic_hist_mean.loc[date, :]
            df_cov_mat = df_cov_mat_series[df_cov_mat_series.index.get_level_values(0) == date]
            #print(f"ICIR with uniform weights: {get_ic_ir(uniform_weights)}")

            def get_ic_ir(factor_weights):
                """
                return the objective function for optimization
                """
                # w.T * IC
                ic_mean = factor_weights.transpose() @ df_ic.values.flatten()
                # w.T * Sigma * w
                ic_var = factor_weights @ df_cov_mat.values @ factor_weights.transpose()
                return ic_mean / (ic_var ** 0.5)

            #optimization step with a constraint that all weights are non-negative
            opt_result = scipy.optimize.minimize(
                            lambda w: -get_ic_ir(w),
                            self.uniform_weights, # initial guess of weights, so just use the uniform
                            bounds=[(0, 1)] * self.num_factors,
                            constraints=({"type": "eq", "fun": lambda weight: np.sum(weight) - 1})
                        )
            #fill the optimized weight values into the container, by row
            self.df_opt_factor_weights.loc[date, :] = list(opt_result.x) + [get_ic_ir(self.uniform_weights), get_ic_ir(opt_result.x)]
            #print(f"ICIR with optimal weights: {get_ic_ir(opt_factor_weight)}")

        #change all values as type float64    
        self.df_opt_factor_weights = self.df_opt_factor_weights.astype('float64')
        return self.df_opt_factor_weights[self.weight_cols]
    
class FactorCombinationWeightedByReturn(FactorCombinator):
    """
    Combines factor exposures according to the weights whose values equal to (half-life) weighted averages of factor returns
    Here, factor returns are referred as the 'coef_series_mean' in class TTester in single_factor.py
    i.e. the coefficient corresponding to the tested single factor in the weighted least square

    For detailed calculation formulas， see Huatai MultiFactor Report #10 华泰金工多因子系列之十：因子合成方法实证分析
    """
    pass

class FactorCombinationWeightedByIC(FactorCombinator):
    """
    Combines factor exposures according to the weights whose values equal to (half-life) weighted averages of factors' RankIC values
    Here, factors' RankIC values are referred as the 'ic_series_mean' in class ICTester in single_factor.py

    For detailed calculation formulas， see Huatai MultiFactor Report #10 华泰金工多因子系列之十：因子合成方法实证分析   
    """
    pass

class FactorCombinationPCA(FactorCombinator):
    """
    Combines factor exposures according to Principle Component Analysis(PCA) and choose the first component as the combinated factor.
    """
    pass