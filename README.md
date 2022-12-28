<p align="center">
    <a href="https://github.com/huggingface/speechbox/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/speechbox.svg">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg">
    </a>
</p>

🤗 Speechbox offers a set of speech processing tools, such as punctuation restoration.

# Installation

**With `pip`** (official package)
    
```bash
pip install speechbox
```

# Tasks

| Task | Description |
|-|-|
| [Punctuation Restoration](#punctuation-restoration) | Punctuation restoration allows one to predict capitalized words as well as punctuation by using [Whisper](https://huggingface.co/models?other=whisper). |

## Punctuation Restoration

Punctuation restoration relies on the premise that [Whisper](https://huggingface.co/models?other=whisper) can understand universal speech. The model is forced to predict the passed words, 
but is allowed to capitalized letters, remove or add blank spaces as well as add punctuation. 
Punctuation is simply defined as the offial Python [string.Punctuation](https://docs.python.org/3/library/string.html#string.punctuation) characters.

### Example

In order to use the punctuation restoration task, you need to install [Transformers](https://github.com/huggingface/transformers):

```
pip install --upgrade transformers
```

For this example, we will additionally make use of [datasets](https://github.com/huggingface/datasets) to load a sample audio file:

```
pip install --upgrade datasets
```

Now we stream a single audio sample, load the punctuation restoring class with ["openai/whisper-tiny.en"](https://huggingface.co/openai/whisper-tiny.en) and add punctuation to the transcription.


```python
from speechbox import PunctuationRestorer
from datasets import load_dataset

streamed_dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)

# get first sample
sample = next(iter(streamed_dataset))

# print out normalized transcript
print(sample["text"])
# => "HE WAS IN A FEVERED STATE OF MIND OWING TO THE BLIGHT HIS WIFE'S ACTION THREATENED TO CAST UPON HIS ENTIRE FUTURE"

# load the restoring class
restorer = PunctuationRestorer.from_pretrained("openai/whisper-tiny.en")
restorer.to("cuda")

restored_text, log_probs = restorer(sample["audio"]["array"], sample["text"], sampling_rate=sample["audio"]["sampling_rate"], num_beams=1)

print("Restored text:\n", restored_text)
```

See [examples/restore](https://github.com/huggingface/speechbox/blob/main/examples/restore.py) for more information.
