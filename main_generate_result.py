import argparse
import numpy as np
from preprocess import *
from train import test
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

def generate_test(cfg_proj, iter, seed = 1):
    text, mmse, dx, paths, lan_detected_train, tkdname_train = get_data(cfg_proj, mode = "train")
    embedding_train, acoustic_train = get_features_train(text, cfg_proj, paths)  

    text, paths, lan_detected_test, tkdname = get_data(cfg_proj, mode = "test")

    embedding_test, acoustic_test = get_features_test(text, cfg_proj, paths)  

    acoustic_all = np.concatenate((acoustic_train, acoustic_test))
    acoustic_all = StandardScaler().fit_transform(acoustic_all)
    acoustic_train = acoustic_all[:len(acoustic_train)]
    acoustic_test = acoustic_all[len(acoustic_train):]
    
    features_train, features_test = np.concatenate([embedding_train, acoustic_train],  axis = -1), np.concatenate([embedding_test, acoustic_test],  axis = -1)
    features_train, features_test = feature_selection(features_train, dx, features_test, cfg_proj.ft_num)
    
    if cfg_proj.task == "Regression":
        test(features_train, mmse, features_test, cfg_proj, paths, iter, seed, lan_detected_train, lan_detected_test, tkdname_train, dx_train = dx)
    else:
        test(features_train, dx, features_test, cfg_proj, paths, iter, seed, lan_detected_train, lan_detected_test, tkdname_train)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  

    parser.add_argument("--embedding-model", type=str, default="bert-base-uncased", required=False) # bert-base-uncased, bert-base-multilingual-uncased
    parser.add_argument("--features", type=str, default = ["embedding", "acoustic"], required=False) # ["embedding", "acoustic"]
    parser.add_argument("--embedding-layer", type = str, default = "last_hidden_state", required = False) #last_hidden_state, pooler_output
    parser.add_argument("--translate", type = bool, default = True, required = False) # True or False
    parser.add_argument("--img_index_sbj", type=int, default = [0,1,2], required=False) #[0,1,2], [0]
    cfg_proj = parser.parse_args()

    classification_hyperparameters = [['bootstrap-lr', 1400, ['MFCC', 'eGeMAPS'], False],  
                                      ['bootstrap-lr', 1300, ['MFCC', 'eGeMAPS'], False], 
                                      ['mlp', 1500, ['MFCC'], False], 
                                      ['mlp', 1200, ['MFCC'], False], 
                                      ['logistic', 1600, ['MFCC'], True]]
    
    regression_hyperparameters = [['svr', 1800, ['MFCC', 'eGeMAPS'], False], 
                                  ['svr', 1500, ['MFCC'], False], 
                                  ['svr', 1800, ['MFCC', 'eGeMAPS'], True], 
                                  ['RandomForest', 1600, ['MFCC'], False, 'mlp'], 
                                  ['RandomForest', 1600, ['MFCC'], False, 'logistic']]

    iter = 0
    for hyperparameter in classification_hyperparameters:
        cfg_proj.task = "Classifier"
        cfg_proj.ft_num = hyperparameter[1]
        cfg_proj.clf = hyperparameter[0]
        cfg_proj.flag_bad_train_filter = hyperparameter[3]
        cfg_proj.acoustic = hyperparameter[2]
        generate_test(cfg_proj, iter = iter+1)
        iter += 1
    
    iter = 0
    for hyperparameter in regression_hyperparameters:
        cfg_proj.task = "Regression"
        cfg_proj.ft_num = hyperparameter[1]
        cfg_proj.reg = hyperparameter[0]
        cfg_proj.acoustic = hyperparameter[2]
        cfg_proj.flag_bad_train_filter = hyperparameter[3]
        if len(hyperparameter) == 5:
            cfg_proj.flag_multi_reg = True
            cfg_proj.clf = hyperparameter[4]
        else:
            cfg_proj.flag_multi_reg = False
        generate_test(cfg_proj, iter = iter+1)
        iter += 1
