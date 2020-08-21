import sys
import pickle
import numpy as np
import pandas as pd
from numpy import savetxt
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier 

def training(original_features,  pca_file_name, trained_file_name):
    original_labels = np.array(original_features["stress_level"])

    original_features= original_features.drop(["stress_level"], axis = 1)
    headers =  original_features.columns.values

    original_features = np.array(original_features)

    # apply oversampling method 
    oversample = SMOTE(k_neighbors = 1)
    original_features, original_labels = oversample.fit_resample(original_features, original_labels)
    
    # apply pca
    pca = PCA(10, random_state = 200)
    original_features = pca.fit_transform(original_features)
    pickle.dump(pca, open(pca_file_name, 'wb'))

    clf = GradientBoostingClassifier(random_state=2, n_estimators = 1001, max_depth=2 )
    clf.fit(original_features,original_labels)

    # save the model to disk
    pickle.dump(clf, open(trained_file_name, 'wb'))

def testing(original_features, pca_file_name, trained_file_name):
    original_features_event = original_features[["event_id"]]
    pca = pickle.load(open(pca_file_name, 'rb'))
    headers =  original_features_event.columns.values
    original_features_pca = pca.transform(original_features)
    loaded_model = pickle.load(open(trained_file_name, 'rb'))
    predictions = loaded_model.predict(original_features_pca)
    predictions = predictions.reshape(len(predictions), 1)
    all = np.concatenate((original_features_event, predictions), axis=1)
    headers= [str(e) for e in headers] + ["stress_level"]
    savetxt("output/predicted_label.csv", all, header =  ", ".join(headers) , delimiter=",", fmt="%d", comments="")
