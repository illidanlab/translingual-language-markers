import glob
import os
import pandas as pd
from tqdm import tqdm
import whisper
import torch
import argparse

parser = argparse.ArgumentParser()  
parser.add_argument("--path_root", type=str, default="/localscratch2/hoangbao/TAUKADIAL-24", required=False)
args = parser.parse_args()

model_id = "large"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
whisper_model = whisper.load_model(model_id, device = device)

for mode in ["test"]:
    paths_wav = sorted(glob.glob(os.path.join(args.path_root, f"{mode}/*.wav")))
    paths_csv = os.path.join(args.path_root, f"{mode}/groundtruth.csv")
    
    if mode == "test":
        df = pd.read_csv(paths_csv, sep = ";")
    else:
        df = pd.read_csv(paths_csv)

    lan_detected_list = []
    lan_text_list = []

    wav_name_list = list(df["tkdname"])
    pbar = tqdm(total = len(wav_name_list))
    for name_wav in wav_name_list:
        path_wav = os.path.join(args.path_root, mode, name_wav)

        result = whisper_model.transcribe(path_wav)

        lan_detected = result["language"]
        lan_text = result["text"]

        if lan_detected not in ["en", "zh"]:
            result = whisper_model.transcribe(path_wav, language = "zh")
            lan_detected = result["language"]
            lan_text = result["text"]

        lan_detected_list.append(lan_detected)
        lan_text_list.append(lan_text)
        pbar.update(1)
    pbar.close()

    df["lan_detected"] = lan_detected_list
    df["lan_transcribe"] = lan_text_list
    df.to_csv(f"{mode}/groundtruth_WhisperLarge.csv", index = False)
