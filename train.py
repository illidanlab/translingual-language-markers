from sklearn.model_selection import train_test_split
from sklearn import linear_model, ensemble
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from preprocess import feature_selection
from mlp_pytorch import MLP
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def perf_metric_classification(Y_test, y_pred):
    conf_matrix = confusion_matrix(Y_test, y_pred)
    try:
        true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()
        specificity = true_negatives / (true_negatives + false_positives)
        sensitivity = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (sensitivity * (specificity / (sensitivity + specificity)))
        accuracy = (specificity + sensitivity) /2
    except:
        f1_score, sensitivity, specificity, accuracy = None, None, None, None
    return f1_score, sensitivity, specificity, accuracy

def perf_metric_regression(Y_test, y_pred):
    mse_ = mean_squared_error(Y_test, y_pred, squared = False)
    r2_ = r2_score(Y_test, y_pred)
    return mse_, r2_

def extract_features(y_train, y_test, lan_train, lan_test, x_train, x_test, mmse_train, mmse_test, target, lang):
    id_train = [id for id, (i, j) in enumerate(zip(y_train, lan_train)) if i == target and j == lang]
    feature_train = x_train[id_train, :]
    mmse_train_filtered = [mmse_train[i] for i in id_train]

    id_test = [id for id, (i, j) in enumerate(zip(y_test, lan_test)) if i == target and j == lang]
    feature_test = x_test[id_test, :]
    mmse_test_filtered = [mmse_test[i] for i in id_test]
    lan_detected_test_filtered = [lan_test[i] for i in id_test]

    return feature_train, mmse_train_filtered, feature_test, mmse_test_filtered, lan_detected_test_filtered

def train(features, mmse, dx, cfg_proj, lan_detected, tkdname, mode = 0):
    # mode = 1 means finding bad subjects; otherwise, just train normally.

    f1, spec, sens, acc, rmse, r2 = [], [], [], [], [], []
    pbar = tqdm(total = cfg_proj.iteration)
    freq = {}

    for iter in range(cfg_proj.iteration):
        X_train, X_test, mmse_train, mmse_test, Y_train, Y_test, lan_detected_train, lan_detected_test, tkdname_train, tkdname_test = train_test_split(features, mmse, dx, lan_detected, tkdname, test_size = 0.1,\
        random_state = iter, stratify = dx)

        if cfg_proj.flag_bad_train_filter and mode == 0:
            threshold_count = 4
            df = pd.read_csv("train/BadSubjects.csv")
            name = list(df["Bad Subject"])
            count = list(df["Frequency"])
            name_f =  [n for n, c in zip(name, count) if c > threshold_count]

            #filter train
            id_keep = [id for id, i in enumerate(tkdname_train) if int(i) not in name_f]
            X_train = X_train[id_keep, :]
            Y_train = [Y_train[i] for i in id_keep]
            lan_detected_train = [lan_detected_train[i] for i in id_keep]
            tkdname_train = [tkdname_train[i] for i in id_keep]
            mmse_train = [mmse_train[i] for i in id_keep]
    
        if cfg_proj.ft_sel: 
            X_train, X_test = feature_selection(X_train, Y_train, X_test, cfg_proj.ft_num)

        if cfg_proj.clf == "logistic":
            clf = linear_model.LogisticRegression(penalty = "l2", dual = False, solver = "liblinear", max_iter = 200, tol = 1e-4, random_state = iter)
            clf.fit(X_train, Y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
        elif cfg_proj.clf == "mlp":
            clf = MLP(seed = iter)
            clf.fit(X_train, Y_train, lan_detected_train)
            y_pred = clf.predict(X_test)
       
        f1_score, sensitivity, specificity, accuracy = perf_metric_classification(Y_test, y_pred)
        f1.append(f1_score)
        sens.append(sensitivity)
        spec.append(specificity)
        acc.append(accuracy)

        if mode == 1:
            for i in range(len(tkdname_test)):
                if y_prob[i][1-Y_test[i]] >= 0.8:
                    if tkdname_test[i] not in freq:
                        freq[tkdname_test[i]] = 0
                    freq[tkdname_test[i]] += 1

        def reg_tasks(feature_train, mmse_train, feature_test):
            if cfg_proj.reg == "svr":
                reg = SVR(C = 8, kernel = "rbf", random_state = iter)
            elif cfg_proj.reg == "RandomForest":
                reg = RandomForestRegressor(n_estimators = 50, random_state = iter)
            reg.fit(feature_train, mmse_train)
            mmse_pred = [min(max(i, 13), 30) for i in reg.predict(feature_test)]
            return mmse_pred
        
        if not cfg_proj.flag_multi_reg:
            y_pred = reg_tasks(X_train, mmse_train, X_test)
            Y_test = mmse_test
            mse_, r2_ = perf_metric_regression(Y_test, y_pred)
            rmse.append(mse_)
            r2.append(r2_)
        else:
            Y_test_from_classifier = y_pred #import which make reg task related to classification task
                        
            # Extract features for all groups
            feature_train_nc_en, mmse_train_nc_en, feature_test_nc_en, mmse_test_nc_en, lan_detected_nc_en = extract_features(Y_train, Y_test_from_classifier, lan_detected_train, lan_detected_test, X_train, X_test, mmse_train, mmse_test, 0, "en")
            feature_train_mci_en, mmse_train_mci_en, feature_test_mci_en, mmse_test_mci_en, lan_detected_mci_en = extract_features(Y_train, Y_test_from_classifier, lan_detected_train, lan_detected_test, X_train, X_test, mmse_train, mmse_test, 1, "en")
            feature_train_nc_zh, mmse_train_nc_zh, feature_test_nc_zh, mmse_test_nc_zh, lan_detected_nc_zh = extract_features(Y_train, Y_test_from_classifier, lan_detected_train, lan_detected_test, X_train, X_test, mmse_train, mmse_test, 0, "zh")
            feature_train_mci_zh, mmse_train_mci_zh, feature_test_mci_zh, mmse_test_mci_zh, lan_detected_mci_zh = extract_features(Y_train, Y_test_from_classifier, lan_detected_train, lan_detected_test, X_train, X_test, mmse_train, mmse_test, 1, "zh")

            # Predict MMSE scores
            mmse_pred_nc_en = reg_tasks(feature_train_nc_en, mmse_train_nc_en, feature_test_nc_en) if len(feature_test_nc_en) > 0 else []
            mmse_pred_mci_en = reg_tasks(feature_train_mci_en, mmse_train_mci_en, feature_test_mci_en) if len(feature_test_mci_en) > 0 else []
            mmse_pred_nc_zh = reg_tasks(feature_train_nc_zh, mmse_train_nc_zh, feature_test_nc_zh) if len(feature_test_nc_zh) > 0 else []
            mmse_pred_mci_zh = reg_tasks(feature_train_mci_zh, mmse_train_mci_zh, feature_test_mci_zh) if len(feature_test_mci_zh) > 0 else []

            mmse_test = mmse_test_nc_en + mmse_test_mci_en + mmse_test_nc_zh + mmse_test_mci_zh
            mmse_pred = mmse_pred_nc_en + mmse_pred_mci_en + mmse_pred_nc_zh + mmse_pred_mci_zh
            mse_, r2_ = perf_metric_regression(mmse_test, mmse_pred)
            rmse.append(mse_)
            r2.append(r2_)

            lan_detected_test = lan_detected_nc_en + lan_detected_mci_en + lan_detected_nc_zh + lan_detected_mci_zh
            Y_test = mmse_test
            y_pred = mmse_pred

        pbar.update(1)
            
    pbar.close()

    if mode == 1:
        df = {"Bad Subject":[], "Frequency":[]}
        for subject in freq:
            df["Bad Subject"].append(subject)
            df["Frequency"].append(freq[subject])
        df = pd.DataFrame(df)
        df.to_csv("train/BadSubjects.csv", index = False)
    else:
        # Print the results
        print("Specificity: %.1f±%.1f"%(np.mean(spec)*100, np.std(spec)*100))
        print("Sensitivity (Recall): %.1f±%.1f"%(np.mean(sens)*100, np.std(sens)*100))
        print("F1 Score: %.1f±%.1f"%(np.mean(f1)*100, np.std(f1)*100))
        print("Accuracy: %.1f±%.1f"%(np.mean(acc)*100, np.std(acc)*100))
        print("R2: %.3f±%.3f"%(np.mean(r2), np.std(r2)))
        print("RMSE: %.3f±%.3f"%(np.mean(rmse), np.std(rmse)))
    
def test(X_train, Y_train, X_test, cfg_proj, paths, iter, seed, lan_detected_train, lan_detected_test, tkdname, dx_train = None):
    if cfg_proj.flag_bad_train_filter:
        threshold_count = 4
        df = pd.read_csv("train/BadSubjects.csv")
        name = list(df["Bad Subject"])
        count = list(df["Frequency"])
        name_f =  [n for n, c in zip(name, count) if c > threshold_count]

        #filter train
        id_keep = [id for id, i in enumerate(tkdname) if int(i) not in name_f]
        X_train = X_train[id_keep, :]
        Y_train = [Y_train[i] for i in id_keep]
        
    if cfg_proj.task == "Classifier":
        if cfg_proj.clf == "logistic":
            clf = linear_model.LogisticRegression(penalty = "l2", dual = False, solver = "liblinear", max_iter = 2000, tol = 1e-4, random_state = seed)
            clf.fit(X_train, Y_train)
            y_pred = clf.predict(X_test)
        elif cfg_proj.clf == "mlp":
            clf = MLP(seed = seed)
            clf.fit(X_train, Y_train, lan_detected_train)
            y_pred = clf.predict(X_test)
        elif cfg_proj.clf == "bootstrap-lr":
            clf = ensemble.BaggingClassifier(estimator=linear_model.LogisticRegression(penalty = "l2", dual = False, solver = "liblinear", max_iter = 200, tol = 1e-4, random_state = seed),
                                            n_estimators=100, 
                                            max_samples = 0.9,
                                            random_state=seed)
            clf.fit(X_train, Y_train)
            y_pred = clf.predict(X_test)
        y_pred_true = []
        for i in range(len(y_pred)):
            y_pred_true.append(y_pred[i])
            y_pred_true.append(y_pred[i])
            y_pred_true.append(y_pred[i])

        y_pred_true = ["NC" if i == 0 else "MCI" for i in y_pred_true]

        df = {"tkdname": paths, "dx":y_pred_true}
        
        with open("Results/taukadial_results_task1_attempt{}.txt".format(iter), 'w') as file:
            file.writelines("tkdname;dx\n")
            for n, y in zip(df["tkdname"], df["dx"]):
                L = "%s;%s\n"%(n, y)
                file.writelines(L)
    else:
        def reg_tasks(feature_train, mmse_train):
            if cfg_proj.reg == "svr":
                reg = SVR(C = 8, kernel = "rbf")
            elif cfg_proj.reg == "RandomForest":
                reg = RandomForestRegressor(n_estimators = 50)
            reg.fit(feature_train, mmse_train)
            return reg
            
        if not cfg_proj.flag_multi_reg:
            reg = reg_tasks(X_train, Y_train)
            y_pred = reg.predict(X_test)
            y_pred = [min(max(i, 13), 30) for i in y_pred]
 
        else:
            if cfg_proj.clf == "logistic":
                clf = linear_model.LogisticRegression(penalty = "l2", dual = False, solver = "liblinear", max_iter = 2000, tol = 1e-4, random_state = seed)
                clf.fit(X_train, dx_train)
                Y_test_from_classifier = clf.predict(X_test)
            elif cfg_proj.clf == "mlp":
                clf = MLP(seed = seed)
                clf.fit(X_train, dx_train, lan_detected_train)
                Y_test_from_classifier = clf.predict(X_test)
            elif cfg_proj.clf == "bootstrap-lr":
                clf = ensemble.BaggingClassifier(estimator=linear_model.LogisticRegression(penalty = "l2", dual = False, solver = "liblinear", max_iter = 200, tol = 1e-4, random_state = seed),
                                                n_estimators=100, 
                                                max_samples = 0.9,
                                                random_state=seed)
                clf.fit(X_train, dx_train)
                Y_test_from_classifier = clf.predict(X_test)

            # Function to train regressors
            def train_regressor(dx_val, lan_val, dx, lan, X, Y):
                ids = [id for id, (i, j) in enumerate(zip(dx, lan)) if i == dx_val and j == lan_val]
                features = X[ids, :]
                targets = [Y[i] for i in ids]
                return reg_tasks(features, targets)

            # Train regressors
            reg_nc_en = train_regressor(0, "en", dx_train, lan_detected_train, X_train, Y_train)
            reg_mci_en = train_regressor(1, "en", dx_train, lan_detected_train, X_train, Y_train)
            reg_nc_zh = train_regressor(0, "zh", dx_train, lan_detected_train, X_train, Y_train)
            reg_mci_zh = train_regressor(1, "zh", dx_train, lan_detected_train, X_train, Y_train)

            y_pred = []
            for i in range(len(X_test)):
                if Y_test_from_classifier[i] == 0 and lan_detected_test[i] == "en":
                    y_pred.append(reg_nc_en.predict(X_test[i:i+1])[0])
                if Y_test_from_classifier[i] == 1 and lan_detected_test[i] == "en":
                    y_pred.append(reg_mci_en.predict(X_test[i:i+1])[0])
                if Y_test_from_classifier[i] == 0 and lan_detected_test[i] == "zh":
                    y_pred.append(reg_nc_zh.predict(X_test[i:i+1])[0])
                if Y_test_from_classifier[i] == 1 and lan_detected_test[i] == "zh":
                    y_pred.append(reg_mci_zh.predict(X_test[i:i+1])[0])   

            y_pred = [min(max(i, 13), 30) for i in y_pred]

        y_pred_true = [item for item in y_pred for _ in range(3)]

        df = {"tkdname": paths, "mmse":y_pred_true}
        
        with open("Results/taukadial_results_task2_attempt{}.txt".format(iter), 'w') as file:
            file.writelines("tkdname;mmse\n")
            for n, y in zip(df["tkdname"], df["mmse"]):
                L = "%s;%f\n"%(n, y)
                file.writelines(L)