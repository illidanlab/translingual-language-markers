import argparse
from preprocess import get_data, get_features
from train import train
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

def main(cfg_proj):
    text, mmse, dx, paths, lan_detected, tkdname = get_data(cfg_proj, mode = "train")
    features = get_features(text, cfg_proj, paths)    
    
    if cfg_proj.flag_bad_train_filter:
        train(features, mmse, dx, cfg_proj, lan_detected, tkdname, mode = 1)

    train(features, mmse, dx, cfg_proj, lan_detected, tkdname, mode = 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  

    parser.add_argument("--embedding-model", type=str, default="bert-base-uncased", required=False) # bert-base-uncased, bert-base-multilingual-uncased
    parser.add_argument("--features", nargs='+', type=str, default = ["embedding", "acoustic"], required=False) # ["embedding",  "acoustic"]
    parser.add_argument("--acoustic", nargs='+', type=str, default = ["MFCC"], required=False) # ["MFCC","GeMAPS","eGeMAPS"]
    parser.add_argument("--iteration", type=int, default = 100, required=False) # embedding
    parser.add_argument("--embedding-layer", type = str, default = "last_hidden_state", required = False) #last_hidden_state, pooler_output
    parser.add_argument("--translate", action='store_true', required = False) # True or False
    parser.add_argument("--clf", type = str, default = "logistic", required = False) # logistic, mlp
    parser.add_argument("--reg", type = str, default = "RandomForest", required = False)  # svr, RandomForest
    parser.add_argument("--ft_sel", action='store_true', required = False)  # Lasso
    parser.add_argument("--ft_num", type = int, default = 1600, required = False) 
    
    #verification func
    parser.add_argument("--flag_bad_train_filter",  action='store_true', required = False) 
    parser.add_argument("--flag_multi_reg", action='store_true', default = True, required = False) 
    parser.add_argument("--img_index_sbj", nargs='+', type=int, default = [0,1,2], required=False) #[0,1,2], [0]

    cfg_proj = parser.parse_args()
    main(cfg_proj)