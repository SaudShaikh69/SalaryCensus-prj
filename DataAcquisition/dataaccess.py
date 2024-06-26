import pandas as pd


class DataAccess:


    def __init__(self):
        self.data_src = r'../Data/adult.csv'

    def access(self):


        try:
            data = pd.read_csv(self.data_src, skipinitialspace=True)
            return data
        except Exception as e:
            raise e


