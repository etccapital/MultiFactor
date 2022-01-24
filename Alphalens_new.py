# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: 'Python 3.9.7 64-bit (''multifactor'': conda)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# using newly pre-processed dataset

# %%
df = pd.read_hdf(DATAPATH + 'stock_factor_data.h5')
df

# %%
pe_ttm = df['PE_TTM'].to_frame()
ps_ttm = df['PS_TTM'].to_frame()
pc_ttm = df['PC_TTM'].to_frame()
pb = df['PB'].to_frame()
pb

# %%
price = df['close'].unstack()
price

# %%
indus = df['pri_indus_code']
indus

# %%
pb_cleaned = al.utils.get_clean_factor_and_forward_returns(factor = pb,
prices = price,
groupby = indus,
binning_by_group=True,
quantiles=5)

# %%
al.tears.create_returns_tear_sheet(pb_cleaned,group_neutral=True,by_group=True,long_short=True)

# %%
al.tears.create_returns_tear_sheet(pb_cleaned)

# %%
al.tears.create_information_tear_sheet(pb_cleaned,group_neutral=True,by_group=True)

# %%
al.tears.create_information_tear_sheet(factor_data=pb_cleaned)

# %%
al.tears.create_turnover_tear_sheet(factor_data=pb_cleaned,turnover_periods=['21D','45D','96D'])
