from datasets import load_dataset
from speechbox import Restorer

from huggingface_hub import login
from huggingface_hub import create_repo
from huggingface_hub import HfApi, CommitOperationAdd

import os
import pandas as pd

REPO_ID = "patrickvonplaten/librispeech_asr_dummy_orthograph"
LOCAL_FOLDER = f"/home/patrick/{REPO_ID.split('/')[-1]}"
LOCAL_FILE = os.path.join(LOCAL_FOLDER, "transcripts.csv")

MODEL_ID = "openai/whisper-tiny.en"

hf_api = HfApi()

create_repo(REPO_ID, exist_ok=True, repo_type="dataset")

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean")["validation"]

restorer = Restorer.from_pretrained(MODEL_ID)
restorer.to("cuda")

def restore(example):
    audio = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]
    text = example["text"].lower()
    restored_text, probs = restorer(audio, text, sampling_rate=sampling_rate)
    return {"orig_transcript": text, "new_transcript": restored_text, "probs": probs}

out = dataset.map(restore, remove_columns=dataset.column_names)

df = pd.DataFrame({'orig_transcript': out["orig_transcript"], 'new_transcript': out["new_transcript"], 'props': out["probs"]})

with open(LOCAL_FILE, "w") as f:
    f.write(df.to_csv(index=False))

operations = [
    CommitOperationAdd(path_in_repo="transcripts.csv", path_or_fileobj=LOCAL_FILE)
]

hf_api.create_commit(
    repo_id=REPO_ID,
    operations=operations,
    commit_message="Upload new transcriptions",
    repo_type="dataset",
)
