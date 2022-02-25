@echo off

if "%~1" == "setup" (
    conda env create -f environment.yml 
)

if "%~1" == "script_to_notebook" (
    cd ..
    move .\scripted_notebook\*.py .\
    move .\notebook\*.ipynb .\
    jupytext --to notebook --update *.py
    move .\*.py .\scripted_notebook\
    move .\*.ipynb .\notebook\
)

if "%~1" == "notebook_to_script" (
    cd ..
    move .\scripted_notebook\*.py .\
    move .\notebook\*.ipynb .\
    jupytext --to py:percent *.ipynb
    move .\*.py .\scripted_notebook\
    move .\*.ipynb .\notebook\
)

pause