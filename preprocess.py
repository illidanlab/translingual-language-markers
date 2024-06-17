import numpy as np
import pandas as pd
from embedding import get_embedding
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

def get_data(cfg_proj, mode = "train"):
    if mode == "train":
        if cfg_proj.translate:
            df = pd.read_csv("train/groundtruth_translate1.csv")
        else:
            df = pd.read_csv("train/groundtruth_WhisperLarge.csv")
    else:
        if cfg_proj.translate:
            df = pd.read_csv("test/groundtruth_translate1.csv")
        else:
            df = pd.read_csv("test/groundtruth_WhisperLarge.csv")

    text = list(df["lan_transcribe"])
    paths = list(df["tkdname"])
    lan_detected = list(df["lan_detected"])
    lan_detected = [lan_detected[i] for i in range(0, len(lan_detected), 3)]

    tkdname = list(df["tkdname"])
    tkdname = [tkdname[i][9:12] for i in range(0, len(tkdname), 3)]

    if mode == "train":
        mmse = list(df["mmse"])
        dx = list(df["dx"])
        dx = [0 if i == "NC" else 1 for i in dx]
        dx = [dx[i] for i in range(0, len(dx), 3)]
        mmse = [mmse[i] for i in range(0, len(mmse), 3)]
        return text, mmse, dx, paths, lan_detected, tkdname
    
    return text, paths, lan_detected, tkdname

def get_features(text, cfg_proj, paths):
    features = None

    embedding_file = cfg_proj.embedding_model.replace("/", "-")+"-"+cfg_proj.embedding_layer+"-"+("Translate" if cfg_proj.translate else "NotTranslate")
    embedding_img = [-1, -1, -1]
    if not os.path.exists("train/%s0.p"%(embedding_file)):
        embedding = get_embedding(text, cfg_proj)
        for i in range(3):
            with open("train/%s%d.p"%(embedding_file, i), 'wb') as file:
                embedding_img[i] = embedding[range(i, len(embedding), 3)]
                pickle.dump(embedding_img[i], file)    
    else:
        for i in range(3):
            with open("train/%s%d.p"%(embedding_file, i), 'rb') as file:
                embedding_img[i] = pickle.load(file)

    embedding_img = np.concatenate([embedding_img[i] for i in cfg_proj.img_index_sbj], axis=-1)    

    with open("train/acoustic.p", 'rb') as file:
        acoustic_all = pickle.load(file)
        
    acoustic_img = None
    for acoustic_type in cfg_proj.acoustic:
        
        acoustic = acoustic_all[acoustic_type]
        if acoustic_type == "MFCC":
            acoustic = np.array([np.concatenate((np.mean(acoustic[path], axis = 1), np.std(acoustic[path], axis = 1),np.var(acoustic[path], axis = 1),\
            np.max(acoustic[path], axis = 1), np.median(acoustic[path], axis = 1),np.min(acoustic[path], axis = 1)), axis = None) for path in paths])
        else:
            acoustic =  np.array([acoustic[path][0] for path in paths])
            
        acoustic_img_sub = [-1, -1, -1]
        for i in range(3):
            acoustic_img_sub[i] = acoustic[range(i, len(acoustic), 3)]
        acoustic_img_sub = np.concatenate([acoustic_img_sub[i] for i in cfg_proj.img_index_sbj], axis=-1)    
        
        if acoustic_img is not None:
            acoustic_img = np.concatenate((acoustic_img, acoustic_img_sub), axis = 1)
        else:
            acoustic_img = acoustic_img_sub

    acoustic_img = StandardScaler().fit_transform(acoustic_img)

    if "embedding" in cfg_proj.features:
        features = embedding_img
 
    if "acoustic" in cfg_proj.features:
        if features is None:
            features = acoustic_img
        else:
            features = np.concatenate([features, acoustic_img],  axis = -1)

    return features

def get_features_train(text, cfg_proj, paths):
    embedding_file = cfg_proj.embedding_model.replace("/", "-")+"-"+cfg_proj.embedding_layer+"-"+("Translate" if cfg_proj.translate else "NotTranslate")
    embedding_img = [-1, -1, -1]
    if not os.path.exists("train/%s0.p"%(embedding_file)):
        embedding = get_embedding(text, cfg_proj)
        for i in range(3):
            with open("train/%s%d.p"%(embedding_file, i), 'wb') as file:
                embedding_img[i] = embedding[range(i, len(embedding), 3)]
                pickle.dump(embedding_img[i], file) 
    else:
        for i in range(3):
            with open("train/%s%d.p"%(embedding_file, i), 'rb') as file:
                embedding_img[i] = pickle.load(file)

    embedding_img = np.concatenate([embedding_img[i] for i in cfg_proj.img_index_sbj], axis=-1)    

    with open("train/acoustic.p", 'rb') as file:
        acoustic_all = pickle.load(file)
    
    acoustic_img = None
    for acoustic_type in cfg_proj.acoustic:
        
        acoustic = acoustic_all[acoustic_type]
        if acoustic_type == "MFCC":
            acoustic = np.array([np.concatenate((np.mean(acoustic[path], axis = 1), np.std(acoustic[path], axis = 1),np.var(acoustic[path], axis = 1),\
            np.max(acoustic[path], axis = 1), np.median(acoustic[path], axis = 1),np.min(acoustic[path], axis = 1)), axis = None) for path in paths])
        else:
            acoustic =  np.array([acoustic[path][0] for path in paths])
            
        acoustic_img_sub = [-1, -1, -1]
        for i in range(3):
            acoustic_img_sub[i] = acoustic[range(i, len(acoustic), 3)]
        acoustic_img_sub = np.concatenate([acoustic_img_sub[i] for i in cfg_proj.img_index_sbj], axis=-1)    
        
        if acoustic_img is not None:
            acoustic_img = np.concatenate((acoustic_img, acoustic_img_sub), axis = 1)
        else:
            acoustic_img = acoustic_img_sub

    return embedding_img, acoustic_img

def get_features_test(text, cfg_proj, paths):
    embedding_file = cfg_proj.embedding_model.replace("/", "-")+"-"+cfg_proj.embedding_layer+"-"+("Translate" if cfg_proj.translate else "NotTranslate")
    embedding_img = [-1, -1, -1]
    if not os.path.exists("test/%s0.p"%(embedding_file)) :
        embedding = get_embedding(text, cfg_proj)
        for i in range(3):
            with open("test/%s%d.p"%(embedding_file, i), 'wb') as file:
                embedding_img[i] = embedding[range(i, len(embedding), 3)]
                pickle.dump(embedding_img[i], file)    
    else:
        for i in range(3):
            with open("test/%s%d.p"%(embedding_file, i), 'rb') as file:
                embedding_img[i] = pickle.load(file)
        
    embedding_img = np.concatenate([embedding_img[i] for i in cfg_proj.img_index_sbj], axis=-1)    
    
    with open("test/acoustic.p", 'rb') as file:
        acoustic_all = pickle.load(file)
        
    acoustic_img = None
    for acoustic_type in cfg_proj.acoustic:
        
        acoustic = acoustic_all[acoustic_type]
        if acoustic_type == "MFCC":
            acoustic = np.array([np.concatenate((np.mean(acoustic[path], axis = 1), np.std(acoustic[path], axis = 1),np.var(acoustic[path], axis = 1),\
            np.max(acoustic[path], axis = 1), np.median(acoustic[path], axis = 1),np.min(acoustic[path], axis = 1)), axis = None) for path in paths])
        else:
            acoustic =  np.array([acoustic[path][0] for path in paths])
            
        acoustic_img_sub = [-1, -1, -1]
        for i in range(3):
            acoustic_img_sub[i] = acoustic[range(i, len(acoustic), 3)]
        acoustic_img_sub = np.concatenate([acoustic_img_sub[i] for i in cfg_proj.img_index_sbj], axis=-1)    
        
        if acoustic_img is not None:
            acoustic_img = np.concatenate((acoustic_img, acoustic_img_sub), axis = 1)
        else:
            acoustic_img = acoustic_img_sub

    return embedding_img, acoustic_img

def feature_selection(data_X, data_Y, data_X_test, ft_num):
    data_Y = np.array(data_Y)
    
    clf = linear_model.Lasso(alpha=0.01, max_iter=10000) 
    clf.fit(data_X, data_Y) 
    vars_weight = clf.coef_

    index_sorted = np.argsort(vars_weight)
    index_selected = index_sorted[-ft_num:]
    
    return data_X[:, index_selected], data_X_test[:, index_selected]