import Dataloader_ricequant as dl
import os

DATAPATH = './data/' # TODO: Considering seperating raw and processed data
stock_path = DATAPATH + 'stock_data/'
csv_names = os.listdir(path=stock_path)
# stock_names = [dl.normalize_code(csv_name.split(".")[0]) for csv_name in csv_names]
