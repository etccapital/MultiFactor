# Multifactor project

## Description

---

## Project environment
To set up the project, first install anaconda and github cli. (Currently only compatible with windows)

1. Open CMD

2. Use `cd` to navtigate to desired folder location

3. Use `git clone https://github.com/etccapital/MultiFactor` to clone the lastest version of the repo

    Note: A login window will pop up to enter credentials for the github 

4. Use `conda env create -f environment.yml` to download configure the all packages needed

5. Use `conda env list` to list all conda packages available. Make sure environment `multifactor` is in the list

6. Download `rq_crendential.json` and save it to root project folder

7. Open `Alphalens_single_factor_testing.ipynb` and run the import cell to make sure all packages are instaslled properly

To inspect packages installed or to make changes:

1. Open CMD

2. Use `conda activate multifactor`

---

## Workflow

1. Download zipped price data and extract them to `.data/price`

2. Download factor data from ricequant with `data_download_and_process.ipynb`

3. Or download zipped factor data and extract them to `.data/factor`

4. (Optional) analyze missing values in price and factor with `data_download_and_process.ipynb`

5. Run `Alphalens_single_factor_testing.ipynb` to examin single factor testing result

---

## Project structure

```
.
├── data/
│   ├── price
│   └── factor
├── Alphalens_single_factor_testing.ipynb
├── data_download_and_process.ipynb
├── dataloader.py (currently_retired)
├── dataloader_ricequant.py
├── Ricequant API.ipynb (To be refactored)
├── environment.yml
├── README.md
└── rq_credential.json (Not commited to repo)
```

Project structure generated with https://tree.nathanfriend.io/