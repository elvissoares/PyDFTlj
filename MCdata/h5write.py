# https://moonbooks.org/Articles/How-to-save-a-large-dataset-in-a-hdf5-file-using-python--Quick-Guide/
import pandas as pd

i = 2

df = pd.read_excel('MCdata-Voukadinova2018.xls',sheet_name='Fig3-Z+=1-rho+=1.0M') 

store = pd.HDFStore('dataVoukadinova2018.hdf5')

store.put('dataset_'+str(i), df)

metadata = {'Z+':1,'Z-':1,'d-(nm)':0.3,'d+(nm)':0.15,'sigma(e/nm2)':-3.125,'c(M)':1.0}

store.get_storer('dataset_'+str(i)).attrs.metadata = metadata

store.close()