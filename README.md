# Multifactor project

## Description

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

## Version Control of .ipynb Files
Currently, we have the following notebooks on our local laptops: `data_download_and_process.ipynb`   `Alphalens_single_factor_testing.ipynb`
However, version control will be impossible if we directly push them to the repo in the form of .ipynb files. This is because jupyter notebooks are               json files and cannot be displayed properly in github. As a result, we will use jupytext(`pip install jupytext --upgrade`) to convert between .ipynb and .py files, and store only .py files in the shared repo. Taking `data_download_and_process.ipynb` as an example, when you finish editing it on your local laptop, run `jupytext --to py:percent data_download_and_process.ipynb` in CMD and the changes will be updated to `data_download_and_process.py`. Then you can merge changes and resolve conflicts in `data_download_and_process.py` as in other python files. To fetch changes from `data_download_and_process.py` to data_download_and_process.ipynb, run `jupytext --to notebook --update data_download_and_process.py` in CMD. Note that the `--update` option is essential as it will only update the code and comments in the .ipynb file while preserving graphs and outputs. \

(On Windows) \
 `./makefiles/makefile_win.bat "script_to_notebook"` to covert scripts to notebooks \
 `./makefiles/makefile_win.bat "notebook_to_script"` to covert notebooks to scripts 

(On Mac) \
 `sh ./makefiles/makefile_mac_py_to_notebook.sh` to covert scripts to notebooks \
 `sh ./makefiles/makefile_mac_notebook_to_py.sh` to covert notebooks to scripts 

 Note: make sure to update the makefile script if more notebooks are added

---

## Workflow

1. Download zipped price data and extract them to `.data/price`

2. Download factor data from ricequant with `data_download_and_process.ipynb`

3. Or download zipped factor data and extract them to `.data/factor`

---

## Project structure

```
.
├── data/
│   ├── price
│   └── factor
├── src/
│   ├── utils.py
│   ├── preprocess.py
│   ├── constants.py
│   └── dataloader.py
├── notebook
├── scripted_notebook/
│   ├── Alphalens_single_factor_testing.py
│   ├── Alphalens_new.py
│   ├── data_download.py
│   ├── single_factor_analysis.py
│   └── factor_combination.py
├── makefile_win.bat
├── environment.yml
├── README.md
└── rq_credential.json (Not commited to repo)
