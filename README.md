# SML_Assignment1
```
git add {notebook name}.ipynb
git commit -m "{commit message}"
git push
```


### To load X (feature array), y (label array) in python
```
from scipy import sparse
temp_sparse_ = sparse.load_npz( "{dataset_name}.npz" )
temp_loaded = temp_sparse_.toarray()
n_samples, n_features = temp_loaded.shape
n_features -= 1 #Since the last column is actually the label
X = temp_loaded[:,:n_features]
y = temp_loaded[:,n_features]
del temp_loaded, temp_sparse_
```


### To save a classifier (or pipelines etc.) just use pickle
```
import pickle as pkl
with open("{filename}", "wb") as writefile:
    pkl.dump( {MLmodel}, writefile )
```

### To load a classifier (or pipelines etc.) just use pickle
```
import pickle as pkl
with open("{filename}", "rb") as loadfile:
    MLmodel = pkl.load( loadfile )
```
