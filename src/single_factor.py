import pandas as pd
from src.utils import *
import scipy.stats
import numpy as np

class TTester:
    def __init__(self):
        self.tval_coef = None
        self.curr_tested_factor = None

    # Get the t-value for all periods
    def run(self, df_backtest: pd.DataFrame, factor_name: str):

        def wls_tval_coef(df):
            #obtain the t-value in WLS of the tested factor
            # 函数内需要用包要额外在这里加上
            import statsmodels.formula.api as smf
            import pandas as pd

            # Weighted Least Square(WLS) uses the square root of market cap of each stock
            # 使用加权最小二乘回归，并以个股流通市值的平方根作为权重
            # other than the factor of interest, we also regress on the industry for neutralization
            # 同时对要测试的因子和行业因子做回归（个股属于该行业为1，否则为0），消除因子收益的行业间差异
            wls_result = smf.wls(formula = f"next_period_return ~ pri_indus_code + {self.curr_tested_factor}", 
                            data=df, weights = df['market_value'] ** 0.5).fit()
            result_tval_coef = pd.Series( {'t_value': wls_result.tvalues.values[0], 'coef': wls_result.params.values[0], 
                                } )
            # result_resid = pd.Series( {'resid': wls_result.resid.values} )
            return result_tval_coef.to_frame().transpose() 

        self.curr_tested_factor = factor_name
        wls_results_tval_coef = applyParallel(df_backtest.groupby(level=0), wls_tval_coef)
        wls_results_tval_coef.index = df_backtest.index.get_level_values(level=0).unique()
        self.tval_coef = wls_results_tval_coef

    def get_summary(self):
        # Get a summary result from the t-value series
        # 回归法的因子评价指标

        # t值序列绝对值平均值
        tval_series_mean = self.tval_coef['t_value'].abs().mean()
        # t 值序列绝对值大于 2 的占比
        large_tval_prop = (self.tval_coef['t_value'].abs() > 2).sum() / self.tval_coef.shape[0]
        # t 值序列均值的绝对值除以 t 值序列的标准差
        standardized_tval = self.tval_coef['t_value'].mean() / self.tval_coef['t_value'].std()
        # 因子收益率序列平均值
        coef_series_mean = self.tval_coef['coef'].mean()
        # 因子收益率均值零假设检验的 t 值
        coef_series_t_val = scipy.stats.ttest_1samp(self.tval_coef['coef'], 0).statistic

        print('t值序列绝对值平均值：', '{:0.4f}'.format(tval_series_mean))
        print('t值序列绝对值大于2的占比：', '{percent:.2%}'.format(percent = large_tval_prop))
        print('t 值序列均值的绝对值除以 t 值序列的标准差：', '{:0.4f}'.format(standardized_tval))
        print('因子收益率均值：', '{percent:.4%}'.format(percent=coef_series_mean))
        print('因子收益率均值零假设检验的 t 值：', '{:0.4f}'.format(coef_series_t_val))

        # Creating a summarizing dataframe
        SUMMARY_ENTRY_NAME = ['t值序列绝对值平均值', 
                            't 值序列绝对值大于 2 的占比', 
                            't 值序列均值的绝对值除以 t 值序列的标准差', 
                            '因子收益率序列平均值', 
                            '因子收益率均值零假设检验的 t 值']

        summary_entry_value = [tval_series_mean, large_tval_prop, standardized_tval, coef_series_mean, coef_series_t_val]

        summary = pd.DataFrame(summary_entry_value, index=SUMMARY_ENTRY_NAME, columns=['value'])

        return summary


class ICTester():
    def __init__(self):
        self.curr_tested_factor = None
        self.ic_series = None
    
    def cross_sectional_ic(self, df):
        return df[['next_period_return', self.curr_tested_factor + '_resid']].corr(method='spearman').iloc[0, 1] 

    def run(self, df_test, factor_name):
        # data preprocess of IC analysis
        # 因子值IC值计算之前的预处理
        # 因子值在去极值、标准化、去空值处理后，在截面期上用其做因变量对市值因子及行业
        # 因子（哑变量）做线性回归，取残差作为因子值的一个替代

        def wls_factor_resid(df):
            import statsmodels.formula.api as smf
            wls_result = smf.wls(formula = f"{self.curr_tested_factor} ~ market_value + pri_indus_code", 
                            data=df).fit()
            return wls_result.resid

        self.curr_tested_factor = factor_name
        factor_resids = applyParallel(df_test.groupby(level=0), wls_factor_resid)
        factor_resids = factor_resids.rename(factor_name + '_resid')

        df_test = df_test.merge(factor_resids, how='left', left_index=True, right_index=True)

        ic_series = df_test.groupby(level=0).apply(self.cross_sectional_ic)

        self.ic_series = ic_series

        return ic_series

    def get_summary(self):
        ic_series_mean = self.ic_series.mean()
        ic_series_std = self.ic_series.std()
        ir = ic_series_mean / ic_series_std
        ic_pos_prop = (self.ic_series > 0).sum() / self.ic_series.shape[0]

        print('IC 均值:','{:0.4f}'.format(ic_series_mean))
        print('IC 标准差:','{:0.4f}'.format(ic_series_std))
        print('IR 比率:','{percent:.2%}'.format(percent=ir))
        print('IC 值序列大于零的占比:','{percent:.2%}'.format(percent=ic_pos_prop))

    def get_graph(self):
        self.ic_series.cumsum().plot()

class HierBackTester():
    def __init__(self):
        pass

    def run(self):
        pass

    def get_summary(self):
        pass

    def get_graph(self):
        pass

class SingleFactorTester():
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.ttester = TTester()
        self.ICtester = ICTester()

    def t_value_test(self, factor_name): 
        self.ttester.run(self.df, factor_name)
        return self.ttester.get_summary()

    def IC_test(self, factor_name):
        self.ICtester.run(self.df, factor_name)
        self.ICtester.get_summary()
        self.ICtester.get_graph()
        
