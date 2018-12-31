import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA

ag_file = '/Users/david/Dropbox/Phate/GMB/AA/ag-cleaned_L6.txt'

df = pd.read_csv(ag_file, sep='\t', header=0)

site = df.ix[:,466].values
counts = df.ix[:,467:].values
l6_names = df.columns.values[467:]

counts_norm = counts / np.sum(counts, axis=1, keepdims=True)
counts_norm = np.sqrt(counts_norm)
counts_norm = counts_norm / np.max(counts_norm)
counts_norm = (counts_norm * 2) - 1

pca = PCA(n_components=100)
Y_pca = pca.fit_transform(counts_norm)

pickle.dump([counts, counts_norm, site, l6_names, Y_pca],
    open( "/Users/david/Dropbox/ARGAN/python/ag/ag.p", "wb" ))


#data = pickle.load(open("/Users/david/Dropbox/ARGAN/python/ag/ag.p","rb"))
