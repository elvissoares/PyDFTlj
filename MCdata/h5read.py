# https://moonbooks.org/Articles/How-to-save-a-large-dataset-in-a-hdf5-file-using-python--Quick-Guide/
import pandas as pd

i = 1
with pd.HDFStore('dataVoukadinova2018.hdf5') as store:
    print(store.info())
    while i:
        data = store['dataset_'+str(i)]
        metadata = store.get_storer('dataset_'+str(i)).attrs.metadata
        print(data)
        print(metadata)
        i+=1