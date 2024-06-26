import pandas as pd
from DataAcquisition.dataaccess import DataAccess


class DataInfo:


    def __init__(self):
        self.data = DataAccess()

    def shape(self):

        try:
            data_shape = self.data.access().shape
            return data_shape
        except Exception as e:
            raise e

    def size(self):

        """

        Method Name : get_size()
        Description : This method is used to find overall size of the dataset loaded.
        output      : Integer Value
        on Failure  : raise exception

        """

        try:
            data_size = self.data.access().size
            return data_size
        except Exception as e:
            raise e

    def info(self):

        """

        Method_Name : get_info()
        Description : This method is used to find the info i.e data type of columns,check null values , values present in column,etc.
        output      : Pandas DataFrame
        on failure  : raise Exception
        
        """

        try:
            data_info = self.data.access().info()
            return data_info
        except Exception as e:
            raise e




