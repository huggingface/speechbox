from datasets import load_dataset

from speechbox import Restorer

restorer = Restorer.from_pretrained("openai/whisper-tiny.en")

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean")[
    "validation"
]
audio = dataset[0]["audio"]["array"]
text = dataset[0]["text"].lower()


restored_text = restorer(audio, text)
