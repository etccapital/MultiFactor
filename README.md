# Multifactor project

![](Multi-Factor%20Workflow.png)

## Description
This is a practical multi-factor backtesting framework from scratch based on Huatai Security's(one of China's largest sell side) financial engineering report, as a part of the quantitative finance research project development in [ETC Investment Group](https://etccapital.ca/). Steps include factor data collection and preprocessing, single factor testing, building return model, building risk model, and result analysis. 

Do not distribute for use without explicit consent from contributing members of ETC Quant. 
---

## Project environment
To set up the project, first install anaconda and github cli. (Currently only compatible with windows)

1. Open CMD/bash

2. `cd` to navtigate to desired folder location

3. `git clone https://github.com/etccapital/MultiFactor` to clone the lastest version of the repo

4. (Linux/MacOS) `conda env create -f environment.yml` to download configure the all packages needed \
   (Windows) `./makefile_win.bat "setup"` to download configure the all packages needed

5. Use `conda env list` to list all conda packages available. Make sure environment `multifactor` is in the list

6. Download `rq_crendential.json` and save it to root project folder

7. Convert target python files into jupyter notebooks. See "Version Control of .ipynb Files" section below. 

To inspect packages installed or to make changes:

1. Open CMD/bash

2. Use `conda activate multifactor`

---

## Workflow

1. Download zipped price data and extract them to `.data/price`

2. Download factor data from ricequant with `data_download_and_process.ipynb`

3. Or download zipped factor data and extract them to `.data/factor`

---

## Project structure
Use command `tree` in command line to generate the following folder structure. 
Whenever you change the folder structure, please update the following diagram and update the corresponding file to the OneDrive folder.
```
.
├── Data
│   ├── factor
│   │   ├── cashflow
│   │   │   ├── cash_flow_per_share_ttm.h5
│   │   │   └── cash_flow_ratio_ttm.h5
│   │   ├── dividend
│   │   ├── financial_quality
│   │   │   ├── debt_to_asset_ratio_ttm.h5
│   │   │   ├── fixed_asset_ratio_ttm.h5
│   │   │   └── return_on_equity_ttm.h5
│   │   ├── growth
│   │   │   └── inc_revenue_ttm.h5
│   │   ├── momentum
│   │   ├── size
│   │   │   └── market_cap_3.h5
│   │   ├── technical
│   │   ├── value
│   │   │   ├── book_to_market_ratio_ttm.h5
│   │   │   ├── ev_ttm.h5
│   │   │   ├── pb_ratio_ttm.h5
│   │   │   ├── pcf_ratio_ttm.h5
│   │   │   ├── pe_ratio_ttm.h5
│   │   │   ├── peg_ratio_ttm.h5
│   │   │   └── ps_ratio_ttm.h5
│   │   └── volatility
│   ├── index_data
│   │   └── sh000300.csv
│   ├── raw_data
│   │   ├── df_basic_info.h5
│   │   ├── industry_mapping.h5
│   │   ├── is_st.h5
│   │   ├── is_suspended.h5
│   │   ├── listed_dates.h5
│   │   ├── stock_names.h5
│   │   ├── rebalancing_dates.h5
│   │   └── industry_code_to_names.xlsx
│   ├── stock_data
│   │   ├── sh600000.csv
│   │   ...
│   │   └── sz301039.csv
├── README.md
├── environment.yml
├── makefiles
│   ├── makefile_mac_notebook_to_py.sh
│   ├── makefile_mac_py_to_notebook.sh
│   └── makefile_win.bat
├── not useful temporarily
│   ├── Dataloader.py
│   └── Ricequant API.ipynb
├── notebook
│   ├── Alphalens_new.ipynb
│   ├── Alphalens_single_factor_testing.ipynb
│   ├── data_download.ipynb
│   ├── data_download_and_process.ipynb
│   ├── factor_combination.ipynb
│   ├── portfolio_optimization.ipynb
│   └── single_factor_analysis.ipynb
├── rq_credential.json
├── scripted_notebook
│   ├── Alphalens_new.py
│   ├── Alphalens_single_factor_testing.py
│   ├── data_download.py
│   ├── data_download_and_process.py
│   ├── factor_combination.py
│   ├── portfolio_optimization.py
│   └── single_factor_analysis.py
└── src
    ├── __init__.py
    ├── constants.py
    ├── dataloader.py
    ├── factor_combinator.py    
    ├── portfolio_optimizer.py
    ├── preprocess.py
    └── utils.py
```  
---

## Version Control of .ipynb Files
Currently, we have the following notebooks on our local laptops: `data_download_and_process.ipynb`   `Alphalens_single_factor_testing.ipynb`
However, version control will be impossible if we directly push them to the repo in the form of .ipynb files. This is because jupyter notebooks are               json files and cannot be displayed properly in github. As a result, we will use jupytext(`pip install jupytext --upgrade`) to convert between .ipynb and .py files, and store only .py files in the shared repo. Taking `data_download_and_process.ipynb` as an example, when you finish editing it on your local laptop, run `jupytext --to py:percent data_download_and_process.ipynb` in CMD and the changes will be updated to `data_download_and_process.py`. Then you can merge changes and resolve conflicts in `data_download_and_process.py` as in other python files. To fetch changes from `data_download_and_process.py` to data_download_and_process.ipynb, run `jupytext --to notebook --update data_download_and_process.py` in CMD. Note that the `--update` option is essential as it will only update the code and comments in the .ipynb file while preserving graphs and outputs. \

To save your time, we have made the following shell scripts/makefiles:

(On Windows) \
 `./makefiles/makefile_win.bat "script_to_notebook"` to covert scripts to notebooks \
 `./makefiles/makefile_win.bat "notebook_to_script"` to covert notebooks to scripts 

(On Mac) \
 `sh ./makefiles/makefile_mac_py_to_notebook.sh` to covert scripts to notebooks \
 `sh ./makefiles/makefile_mac_notebook_to_py.sh` to covert notebooks to scripts 

 Note: make sure to update the makefile script if more notebooks are added
