import numpy as np
import sys
import json
import warnings
import time
from itertools import product
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import make_pipeline
from scipy import sparse

def main():
    job_id = int(sys.argv[1]) #Spartan jobid for parallel jobs
    
    #Load features and labels into X, y
    temp_sparse_ = sparse.load_npz( "domain2_X_y_csr.npz" )
    temp_loaded = temp_sparse_.toarray()
    n_samples, n_features = temp_loaded.shape
    n_features -= 1 #Since the last column is actually the label
    X = temp_loaded[:,:n_features]
    y = temp_loaded[:,n_features]
    del temp_loaded, temp_sparse_
    
    #Use adjusted balanced accuracy with value 0 for randomly guessing even on imbalanced data.
    def evaluation(estimator, X, y):
        y_pred = estimator.predict(X)
        return balanced_accuracy_score( y, y_pred, adjusted=True )
    
    ratios = np.arange(0.3, 1.0, 0.1) #Ratio after over/under sampling.
    k_neighbors_possiblevals = np.arange(3,20,1)
    n_neighbors_enn_possiblevals = np.arange(3,20,1)
    percentile = 20 #Use only a few features for faster runtimes.
    params = {}
    for i, param in enumerate(product( ratios,
                                        k_neighbors_possiblevals,
                                        n_neighbors_enn_possiblevals
                                    )):
        params[i] = param
    param = params[job_id]
    
    #Build pipeline model with selected hyperparameters
    selector = SelectPercentile( chi2, percentile=percentile )
    sampling_strategy_smote, k_neighbors, n_neighbors_enn = param
    smote = SMOTE(sampling_strategy=sampling_strategy_smote,
                    random_state=0,
                    k_neighbors=k_neighbors,
                    n_jobs=1)
    enn = EditedNearestNeighbours(n_neighbors=n_neighbors_enn, n_jobs=1)
    resampler = SMOTEENN(random_state=0, smote=smote, enn=enn)
    estimator = LogisticRegressionCV( scoring = evaluation, n_jobs=1 )
    model = make_pipeline(selector, resampler, estimator )
    
    #Train and cross validate model
    cv = StratifiedKFold( n_splits=5, shuffle=True, random_state=job_id+2024 )
    scores = []
    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        t1 = time.perf_counter()
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
        model.fit(X_train, y_train)
        score = evaluation(model, X_test, y_test)
        scores += [ score ]
        t2 = time.perf_counter()
        total_time = t2 - t1
        print(f"Fold {i}: {score.round(3)} ({round(total_time,2)}s)")
        
    with open("spartan/{}.txt".format(job_id), "w") as f:
        json.dump( scores, f )
    
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()