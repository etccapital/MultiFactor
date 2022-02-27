import pandas as pd
import pathos
import multiprocessing
import time

def timer(func):
    # This decorator function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def sort_index_and_col(df) -> pd.DataFrame:
    # sort the dataframe by index and column
    return df.sort_index(axis=0).reindex(sorted(df.columns), axis=1)
    # the following might achieve the same result in a cleaner way
    # return df.sort_index(axis=0).sort_index(axis=1)

def applyParallel(dfGrouped, func):
    # parrallel computing version of pd.groupby.apply, works most of the time but not always
    # I mainly use it for cases where func takes in a dataframe and outputs a dataframe or a series
    with pathos.multiprocessing.ProcessPool(pathos.helpers.cpu_count()) as pool:
        ret_list = pool.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)

def remove_outlier(df, n=3):
    # for any factor, if the stock's factor exposure lies more than n times MAD away from the factor's median, 
    # reset that stock's factor exposure to median + n * MAD/median - n* MAD
    med = df.median(axis=0)
    MAD = (df - med).abs().median()
    upper_limit = med + n * MAD
    lower_limit = med - n * MAD
    # print(f"lower_limit = {lower_limit}, upper_limit = {upper_limit}")
    # pd.DataFrame.where replaces data in the dataframe by 'other' where the condition is False
    df = df.where(~( ( df > upper_limit) & df.notnull() ), other = upper_limit, axis=1)
    df = df.where(~( ( df < lower_limit) & df.notnull() ), other = lower_limit, axis=1)

    #is_outlier is a boolean dataframe indicating if each entry is an outlier or not
    is_outlier = (df > upper_limit) | (df < lower_limit)
    #check that there are no more outliers
    assert(is_outlier.sum().sum() == 0)
    return df

def standardize(df):
    # on each rebalancing date, each standardized factor has mean 0 and std 1
    return (df - df.mean()) / df.std()
