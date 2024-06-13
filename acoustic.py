import librosa
import librosa.display
import glob
import numpy as np
import os
import pickle
from tqdm import tqdm
import opensmile
from multiprocessing import Pool
import multiprocessing
import argparse

# Extract OPEN SMILE
def extract_opensmile(path, features = "gemaps"):
    if features == "gemaps":
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    else:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    y = list(np.array(smile.process_file(path)))
    return y

# MFCC conversion
def extract_mfcc(path):
    '''log spectrogram -> mfcc features
    mfcc: original 13 frequencies
    delta_mfcc, delta2_mfcc: vel. and accel. features
    M: 39 mfcc, delta, delta2 features'''
    y, sr = librosa.load(path)
    S = librosa.feature.melspectrogram(y = y, sr = sr, n_mels = 128)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    return M

parser = argparse.ArgumentParser()  
parser.add_argument("--path_root", type=str, default="/localscratch2/hoangbao/TAUKADIAL-24", required=False)
args = parser.parse_args()

for mode in ["train", "test"]:
    paths = sorted(glob.glob(os.path.join(args.path_root, f"{mode}/*.wav")))

    mfcc_features = {}
    gemaps_features = {}
    egemaps_features = {}

    def func(path):
        mfcc_f = extract_mfcc(path)
        gemaps_f = extract_opensmile(path, features = "gemaps")
        egemaps_f = extract_opensmile(path, features = "egemaps")
        return os.path.basename(path), mfcc_f, gemaps_f, egemaps_f

    with Pool(multiprocessing.cpu_count()) as p:
        pbar = tqdm(total = len(paths))
        for n, mfcc_f, gemaps_f, egemaps_f in p.imap_unordered(func, paths):
            mfcc_features[n] = mfcc_f
            gemaps_features[n] = gemaps_f
            egemaps_features[n] = egemaps_f
            pbar.update(1)
        pbar.close()

    features = {
        "MFCC": mfcc_features,
        "GeMAPS": gemaps_features,
        "eGeMAPS": egemaps_features,
    }

    with open(f"{mode}/acoustic.p", "wb") as file:
        pickle.dump(features, file)    
