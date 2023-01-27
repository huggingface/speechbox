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

With `pip` (official package)
    
```bash
pip install speechbox
```

# Contributing

We ❤️  contributions from the open-source community! 
If you want to contribute to this library, please check out our [Contribution guide](https://github.com/huggingface/speechbox/blob/main/CONTRIBUTING.md).
You can look out for [issues](https://github.com/huggingface/speechbox/issues) you'd like to tackle to contribute to the library.
- See [Good first issues](https://github.com/huggingface/speechbox/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) for general opportunities to contribute
- See [New Task](https://github.com/huggingface/speechbox/labels/New%20Task) for more advanced contributions. Make sure to have read the [Philosophy guide](https://github.com/huggingface/speechbox/blob/main/CONTRIBUTING.md#philosophy) to succesfully add a new task.

Also, say 👋 in our public Discord channel <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a> under **ML for Audio and Speech**. We discuss the new trends about machine learning methods for speech, help each other with contributions, personal projects or
just hang out ☕.

# Tasks

| Task | Description | Author |
|-|-|-|
| [Punctuation Restoration](#punctuation-restoration) | Punctuation restoration allows one to predict capitalized words as well as punctuation by using [Whisper](https://huggingface.co/models?other=whisper). | [Patrick von Platen](https://github.com/patrickvonplaten) |
| [ASR With Speaker Diarization](#asr-with-speaker-diarization) | Transcribe long audio files, such as meeting recordings, with speaker information (who spoke when) and the transcribed text. | [Sanchit Gandhi](https://github.com/sanchit-gandhi) |

## Punctuation Restoration

Punctuation restoration relies on the premise that [Whisper](https://huggingface.co/models?other=whisper) can understand universal speech. The model is forced to predict the passed words, 
but is allowed to capitalized letters, remove or add blank spaces as well as add punctuation. 
Punctuation is simply defined as the offial Python [string.Punctuation](https://docs.python.org/3/library/string.html#string.punctuation) characters.

**Note**: For now this package has only been tested with:
- [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en)
- [openai/whisper-base.en](https://huggingface.co/openai/whisper-base.en)
- [openai/whisper-small.en](https://huggingface.co/openai/whisper-small.en)
- [openai/whisper-medium.en](https://huggingface.co/openai/whisper-medium.en)

and **only** on some 80 audio samples of [patrickvonplaten/librispeech_asr_dummy](https://huggingface.co/datasets/patrickvonplaten/librispeech_asr_dummy).

See some transcribed results [here](https://huggingface.co/datasets?other=speechbox_punc).

### Web Demo

If you want to try out the punctuation restoration, you can try out the following 🚀 Spaces:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/speechbox/whisper-restore-punctuation)

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

## ASR With Speaker Diarization
Given an unlabelled audio segment, a speaker diarization model is used to predict "who spoke when". These speaker 
predictions are paired with the output of a speech recognition system (e.g. Whisper) to give speaker-labelled 
transcriptions.

The combined ASR + Diarization pipeline can be applied directly to long audio samples, such as meeting recordings, to 
give fully annotated meeting transcriptions. 

### Web Demo

If you want to try out the ASR + Diarization pipeline, you can try out the following Space:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/speechbox/whisper-speaker-diarization)

### Example

In order to use the ASR + Diarization pipeline, you need to install 🤗 [Transformers](https://github.com/huggingface/transformers) 
and [pyannote.audio](https://github.com/pyannote/pyannote-audio):

```
pip install --upgrade transformers pyannote.audio
```

For this example, we will additionally make use of 🤗 [Datasets](https://github.com/huggingface/datasets) to load a sample audio file:

```
pip install --upgrade datasets
```

Now we stream a single audio sample, pass it to the ASR + Diarization pipeline, and return the speaker-segmented transcription:

```python
import torch
from speechbox import ASRDiarizationPipeline
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipeline = ASRDiarizationPipeline.from_pretrained("openai/whisper-tiny", device=device)

# load dataset of concatenated LibriSpeech samples
concatenated_librispeech = load_dataset("sanchit-gandhi/concatenated_librispeech", split="train", streaming=True)
# get first sample
sample = next(iter(concatenated_librispeech))

out = pipeline(sample["audio"])
print(out)
```
