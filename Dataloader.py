import pandas as pd
from datetime import datetime, timedelta

def weekday_between(start: str, end: str):
    """A helper function that returns an array of all weekdays between start and end(exclusive)

    Args:
        start: 
            starting date of desired period, formatted as 'yyyy-mm-dd'
        end: 
            end date of desired period, formatted as 'yyyy-mm-dd'

    Return:
        An array that contains all weekdays between start and end in desired format
    """
    start_obj = datetime.strptime(start, '%Y-%m-%d')
    end_obj = datetime.strptime(end, '%Y-%m-%d')
    date = []
    while start_obj < end_obj:
        # Ignore date if it is a weekend (weekday() returns 0 for Monday and 6 for Sunday)
        if start_obj.weekday() < 5:
            date.append(start_obj.strftime('%Y-%m-%d'))
        start_obj += timedelta(days=1)

    return date

class Dataloader:
    """A class that abstracts the loading process of factor and return data

    Attributes:
        data_path: 
            relative path to the data folder, e.g. './data'
    """
    def __init__(self, data_path: str):
        """Initialize Dataloader with data_path, the relative path to the data folder"""
        self.data_path = data_path

    def load_factor(self, factor_name: str, col_name: list, start_date: str, end_date: str) -> pd.DataFrame:
        """Load specified factor data from local csv files and convert it into a multiindexed dataframe

        Args:
            factor_name: 
                name of factor that is used to structure the folders 
                in the data folder, e.g. 'Beta', 'Momentum'
            col_name: 
                a list of columns to be included in the dataframe, 
                should be a list, e.g. ['beta_1m', 'beta_12m']
            start_date: 
                starting date of desired period for factor data, formatted as 'yyyy-mm-dd'
            end_date: 
                end date of desired period for factor data, formatted as 'yyyy-mm-dd'

        Return:
            A dataframe object with multiindex (date, asset) and columns from factor data files
        """
        factor_data = pd.DataFrame()
        for tdate in weekday_between(start_date, end_date):
            try:
                curr_day = pd.read_csv(self.data_path + factor_name +f'/{tdate}.csv')[col_name]
                curr_day['date'] = tdate
                curr_day['asset'] = curr_day.index.values
                factor_data = factor_data.append(curr_day)
            except:
                # raise Exception(f"IOError for file {str(tdate)}")
                continue

        return factor_data.set_index(['date', 'asset'])

    def load_return(self, ret_filename: str) -> pd.DataFrame:
        """ Load return data from specified file and convert it to a multiindexed serie

        Args: 
            ret_filename: name of the return file used in data folder, e.g. 'daily ret.csv'

        Return:
            A pandas dataframe object with multiindex (date, asset)
        """
        daily_ret = pd.read_csv(self.data_path + ret_filename)
        daily_ret = daily_ret.set_index('Trddt').stack()
        daily_ret = daily_ret.rename_axis(['date', 'asset'])

        return daily_ret.to_frame("1D")

    def load_industry(self, industry_filename: str) -> pd.DataFrame:
        pass
        



