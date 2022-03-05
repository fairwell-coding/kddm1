from helper import unpack_dataset
from helper import read_xls_file
import logging


def main():
    logging.basicConfig(level=logging.DEBUG)

    unpack_dataset("jester_dataset_1_1.zip")
    unpack_dataset("jester_dataset_1_2.zip")
    unpack_dataset("jester_dataset_1_3.zip")

    jester_1 = read_xls_file("jester-data-1.xls")
    jester_2 = read_xls_file("jester-data-2.xls")
    jester_3 = read_xls_file("jester-data-3.xls")


if __name__ == '__main__':
    main()
