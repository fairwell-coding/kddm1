import zipfile
import logging
from os import path
import pandas as pd

data_path_ = "../data/"


def read_xls_file(file_name):
    if path.exists(data_path_ + file_name[:-3] + "csv"):
        df = pd.read_csv(data_path_ + file_name[:-3] + "csv", header=None, low_memory=False)
        return df, df.to_numpy()
    if path.exists(data_path_ + file_name):
        df = pd.read_excel(data_path_ + file_name, engine="xlrd")
        df.to_csv(data_path_ + file_name[:-3] + "csv", index=False)
        logging.info(" Read: " + file_name + " into memory")
        df = pd.read_csv(data_path_ + file_name[:-3] + "csv", header=None, low_memory=False)
        return df, df.to_numpy()


def unpack_dataset(file_name):
    if path.exists(data_path_ + file_name):
        zip_file = zipfile.ZipFile(data_path_ + file_name, 'r')
        zipped_files_name = zip_file.namelist()
        for file in zipped_files_name:
            if path.exists(data_path_ + file):
                logging.info(" Already unpacked: " + file)
            else:
                zip_file.extract(file, data_path_)
                logging.info(" Unpacked: " + file)


