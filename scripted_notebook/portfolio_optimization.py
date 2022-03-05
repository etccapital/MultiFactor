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
print(os.getcwd())

# %%
from src.portfolio_optimizer import *


# %%
def check_symmetric_semi_pos_def(matrix, rtol=1e-05, atol=1e-08):
    # The matrix should be the same as its transpose
    assert( np.allclose(matrix, matrix.T, rtol=rtol, atol=atol) )
    # All eigenvalues should be non-negative. Added absolute tolerance here to allow for rounding errors.
    assert( all(np.linalg.eigvals(matrix) >= (0 - atol) ) )


# %%
df_basic_info = dl.load_basic_info()
filter = preprocess.TimeAndStockFilter(df_basic_info)
df_basic_info = filter.run()

# %%
style_factor_dict = {'value': ['ev_ttm', 'pe_ratio_ttm', 'pb_ratio_ttm', 'peg_ratio_ttm', 'book_to_market_ratio_ttm', 'pcf_ratio_ttm', 'ps_ratio_ttm']}
p = PortfolioOptimizer(df_basic_info, style_factor_dict=style_factor_dict, gamma=2.5)
p.run()
