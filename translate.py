from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import pandas as pd
from tqdm import tqdm

for mode in ["train", "test"]:
    paths = [f"{mode}/groundtruth_WhisperLarge.csv", f"{mode}/groundtruth_translate.csv", f"{mode}/groundtruth_translate1.csv"]

    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="zh")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

    df = pd.read_csv(paths[0])

    text = list(df[df["lan_detected"] == "zh"]["lan_transcribe"])
    translate_text = []

    for t in tqdm(text, desc = "Translate Chinese to English"):
        encoded_zh = tokenizer(t, return_tensors="pt")
        generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
        translate_text += list(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

    df.loc[df["lan_detected"] == "zh", "lan_transcribe"] = translate_text
    df.to_csv(paths[1], index = False)

    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="en")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

    df = pd.read_csv(paths[1])

    text = list(df[df["lan_detected"] == "en"]["lan_transcribe"])
    translate_text = []

    for t in tqdm(text, desc = "Translate English to Chinese"):
        encoded_zh = tokenizer(t, return_tensors="pt")
        generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("zh"))
        translate_text += list(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

    df.loc[df["lan_detected"] == "en", "lan_transcribe"] = translate_text
    df.to_csv(paths[2], index = False)

    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="zh")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

    df = pd.read_csv(paths[2])

    text = list(df[df["lan_detected"] == "en"]["lan_transcribe"])
    translate_text = []

    for t in tqdm(text, desc = "Translate Chinese back to English"):
        encoded_zh = tokenizer(t, return_tensors="pt")
        generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
        translate_text += list(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

    df.loc[df["lan_detected"] == "en", "lan_transcribe"] = translate_text
    df.to_csv(paths[2], index = False)