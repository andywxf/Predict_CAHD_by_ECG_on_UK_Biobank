
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import Lasso,LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from scipy.stats.mstats import winsorize
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras

GPU_num = 7
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
if gpus:
    tf.config.experimental.set_visible_devices(devices=(gpus[GPU_num]), device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[GPU_num], True)

diease_name = "I251"
datafolder = "your_path/"

def divide_train_val_by_kfold(k_fold, fold_sum, X_all,Y_all):
    num_val_samples = int(X_all.shape[0] // fold_sum)
    X_val = X_all[num_val_samples * k_fold:num_val_samples * (k_fold + 1)]
    X_train = pd.concat((X_all[:num_val_samples * k_fold], X_all[num_val_samples * (k_fold + 1):]))#,axis =0 ,axis =0
    y_val = Y_all[num_val_samples * k_fold:num_val_samples * (k_fold + 1)]
    y_train = pd.concat((Y_all[:num_val_samples * k_fold], Y_all[num_val_samples * (k_fold + 1):]))
    return (X_train, y_train, X_val, y_val)

def Save_CV_predict_result(X_val, y_val,  y_pred, Net_id):
    print("X_val: ", X_val.shape)
    X_info = X_val.iloc[:,-2:].copy()
    X_info["Label"] = y_val.values
    X_info["PredictLabel"] = y_pred
    if (os.path.exists(datafolder+"CV_result/"+Net_name[Net_id]+ "_cv.csv")):
        X_exist =  pd.read_csv(datafolder +"CV_result/"+Net_name[Net_id]+"_cv.csv", delimiter=",", index_col = 0,header = 0)
        X_exist = pd.concat((X_exist, X_info))
        X_exist.to_csv(datafolder +"CV_result/"+Net_name[Net_id]+"_cv.csv")
    else:
        X_info.to_csv(datafolder+"CV_result/"+Net_name[Net_id]+"_cv.csv")

def get_dataset():
    feature_name = datafolder+"I251_manual_feature.csv"
    dataset = pd.read_csv(feature_name, delimiter=",", index_col=0, header=0)
    X = dataset.iloc[:, 0:-1]
    X_temp = X.values.T
    for i in range(0, X_temp.shape[0]-2):
        X_temp[i,:] = winsorize(X_temp[i,:], limits=[0.05, 0.05]).data
    X_temp = X_temp.T
    X_temp = pd.DataFrame(X_temp,X.index)
    X_temp.columns = X.columns
    X = X_temp
    Y = dataset.iloc[:, -1]
    Y = Y.astype(int)

    np.random.seed(1000)
    permutation = list(np.random.permutation(X.shape[0]))
    X_shuf = X.iloc[permutation, :]
    Y_shuf = Y.iloc[permutation]

    return X_shuf, Y_shuf

def Xgboost_model(X_train, y_train, X_val, y_val, Net_id):
    X_val_copy = X_val.copy()

    X_train = X_train.iloc[:,0:-2]
    X_val = X_val.iloc[:,0:-2]

    parameters = {
        'max_depth': 7,
        'learning_rate':0.1,
        'n_estimators': 200,
        'min_child_weight':0.01,
        'gamma': 0.3,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
    }
    model = XGBClassifier(**parameters)

    eval_set = [X_val, y_val]
    model.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc",eval_set=[eval_set],verbose=False)

    results = []
    y_pred = model.predict_proba(X_val)
    result = metrics.roc_auc_score(y_val, y_pred[:,1])
    results.append(result)
    result = f1_score(y_val, y_pred[:,1].round(), average='micro')
    results.append(result)
    Save_CV_predict_result(X_val_copy, y_val, y_pred[:,1], Net_id)

    return results

def DecisionTree(X_train, y_train, X_val, y_val, Net_id):
    X_val_copy = X_val.copy()

    X_train = X_train.iloc[:,0:-2]
    X_val = X_val.iloc[:,0:-2]

    mean = X_train.mean(axis=0)
    X_train -= mean
    std = X_train.std(axis=0)
    X_train /= std

    mean = X_val.mean(axis=0)
    X_val -= mean
    std = X_val.std(axis=0)
    X_val /= std

    model = DecisionTreeClassifier(splitter="best",max_features="auto")
    model.fit(X_train, y_train)

    results = []
    y_pred = model.predict_proba(X_val)
    result = metrics.roc_auc_score(y_val, y_pred[:,1])
    results.append(result)
    result = f1_score(y_val, y_pred[:,1].round(), average='micro')
    results.append(result)
    Save_CV_predict_result(X_val_copy, y_val, y_pred, Net_id)
    return results

def SVM_model(X_train, y_train, X_val, y_val, Net_id):
    X_val_copy = X_val.copy()

    X_train = X_train.iloc[:,0:-2]
    X_val = X_val.iloc[:,0:-2]

    mean = X_train.mean(axis=0)
    X_train -= mean
    std = X_train.std(axis=0)
    X_train /= std

    mean = X_val.mean(axis=0)
    X_val -= mean
    std = X_val.std(axis=0)
    X_val /= std

    model = SVC(kernel="linear",   gamma='auto',probability=True)
    model.fit(X_train, y_train)
    results = []
    y_pred = model.predict(X_val)
    result = metrics.roc_auc_score(y_val, y_pred)
    results.append(result)
    result = f1_score(y_val, y_pred.round(), average='micro')
    results.append(result)
    Save_CV_predict_result(X_val_copy, y_val, y_pred, Net_id)
    return results

def RandomForest_model(X_train, y_train, X_val, y_val, Net_id):
    X_val_copy = X_val.copy()
    X_train = X_train.iloc[:,0:-2]
    X_val = X_val.iloc[:,0:-2]

    mean = X_train.mean(axis=0)
    X_train -= mean
    std = X_train.std(axis=0)
    X_train /= std

    mean = X_val.mean(axis=0)
    X_val -= mean
    std = X_val.std(axis=0)
    X_val /= std
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    results = []
    y_pred = model.predict_proba(X_val)
    result = metrics.roc_auc_score(y_val, y_pred[:,1])
    results.append(result)
    result = f1_score(y_val, y_pred[:,1].round(), average='micro')
    results.append(result)
    Save_CV_predict_result(X_val_copy, y_val, y_pred, Net_id)
    return results

def LogRegres_model(X_train, y_train, X_val, y_val, Net_id):
    X_val_copy = X_val.copy()
    X_train = X_train.iloc[:,0:-2]
    X_val = X_val.iloc[:,0:-2]

    model = LogisticRegression(penalty='l2') #
    model.fit(X_train, y_train)
    results = []
    y_pred = model.predict_proba(X_val)
    result = metrics.roc_auc_score(y_val, y_pred[:,1])
    results.append(result)
    result = f1_score(y_val, y_pred[:,1].round(), average='micro')
    results.append(result)
    Save_CV_predict_result(X_val_copy, y_val, y_pred.round(), Net_id)
    return results

def run_model_in_k_fold(k_fold_sum, fold_index, Net_Id):
    (X_train, y_train, X_val, y_val) = divide_train_val_by_kfold(fold_index, k_fold_sum, X_all, Y_all)
    if Net_Id == 0:
        results = Xgboost_model(X_train, y_train, X_val, y_val, Net_Id)
    elif Net_Id == 1:
        results = RandomForest_model(X_train, y_train, X_val, y_val, Net_Id)
    elif Net_Id == 2:
        results = DecisionTree(X_train, y_train, X_val, y_val, Net_Id)
    elif Net_Id == 3:
        results = SVM_model(X_train, y_train, X_val, y_val, Net_Id)
    elif Net_Id == 4:
        results = LogRegres_model(X_train, y_train, X_val, y_val, Net_Id)
    return results


k_fold_sum = 5
Net_name = ["Xgboost","RandForest","DecisionTree", "SVM", "LogRegre"]

for method_id in range(0,5):
    X_all, Y_all = get_dataset()
    Net_Id = method_id
    for fold_index in range(0,k_fold_sum):
        keras.backend.clear_session()
        print(fold_index, k_fold_sum)
        results = run_model_in_k_fold(k_fold_sum, fold_index,Net_Id)
        file_proce = open(datafolder+"/output/ML_auc"+".txt", "a")
        pro_loss = str(Net_name[Net_Id]+" k_flod_num: " +str(fold_index) + ",AUC_F1:" + str(results) + "\n")
        file_proce.write(pro_loss)
        file_proce.close()
