@echo off

if "%~1" == "setup" (
    conda env create -f environment.yml 
)

if "%~1" == "script_to_notebook" (
    jupytext --to notebook --update data_download.py Alphalens_new.py Alphalens_single_factor_testing.py single_factor_analysis.py
)

if "%~1" == "notebook_to_script" (
    jupytext --to py:percent data_download.ipynb Alphalens_new.ipynb Alphalens_single_factor_testing.ipynb single_factor_analysis.ipynb
)

pause