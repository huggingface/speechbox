#!/usr/bin/env python3
import string
from typing import List, Union, Optional

import numpy as np
import torch
import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class Restorer:
    def __init__(self, model: WhisperForConditionalGeneration, processor: WhisperProcessor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model = model
        self.punctuation = self.get_punctuation_tokens()

    def to(self, device):
        self.model = self.model.to(device)

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path):
        model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_or_path, low_cpu_mem_usage=True)
        processor = WhisperProcessor.from_pretrained(pretrained_model_or_path)
        return cls(model, processor)

    def get_punctuation_tokens(self):
        punctuation_tokens = []
        for i in tqdm.tqdm(range(len(self.tokenizer))):
            if self.tokenizer.convert_ids_to_tokens(i) in list(string.punctuation):
                punctuation_tokens.append(i)
        return punctuation_tokens

    def convert_words(self, words):
        word_tokens = []
        for word in words:
            tokens = self.tokenizer.tokenize(word)
            word_tokens.append(self.tokenizer.convert_tokens_to_ids(tokens))

        return word_tokens

    def get_ranks(self, top_k, ids):
        ranks = [(top_k == i).nonzero() if i in top_k else float("inf") for i in ids]
        return ranks

    def __call__(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        transcript: Union[str, List[str]],
        sampling_rate: Optional[int] = None,
    ):
        device = self.model.device
        # there are three possibilities of how a fully orthographic version of `transcript` can 
        # look like:
        # 1) All lower-cased, e.g. "hello"
        # 2) The first letter upper-cased, e.g. "Hello"
        # 3) All upper-cased, e.g. "HELLO"
        # Option 3) is less likely but can occur.
        lower_words = self.convert_words([f" {word.lower()}" for word in transcript.split()])
        upper_words = self.convert_words([f" {word.lower().capitalize()}" for word in transcript.split()])
        all_upper_words = self.convert_words([f" {word.upper()}" for word in transcript.split()])

        # Quick quality check, lower-cased words have to be identical when decoding
        flat_lower_words = [item for sublist in lower_words for item in sublist]
        assert self.tokenizer.decode(flat_lower_words)[1:] == transcript, f"Decoding of {transcript} is wrong."

        # Get encoder hidden states
        input_features = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt")["input_features"]
        input_features = input_features.to(device)

        with torch.no_grad():
            encoder_hidden_states = self.model.model.encoder(input_features).last_hidden_state

        # Define `decoder_start_ids` as initial `current_ids`; they are quite specific to whisper-{}.en checkpoints
        decoder_start_ids = [self.model.config.decoder_start_token_id]
        if self.model.config.forced_decoder_ids is not None:
            for token_id in self.model.config.forced_decoder_ids:
                decoder_start_ids.append(token_id[-1])

        decoder_start_ids = torch.tensor(decoder_start_ids, dtype=torch.long, device=device)[None, :]
        current_ids = decoder_start_ids.broadcast_to(encoder_hidden_states.shape[:1] + decoder_start_ids.shape[1:])

        # loop over words of possibilities 1), 2) and 3).
        for i in range(len(lower_words)):
            tokens_list = (lower_words[i], upper_words[i], all_upper_words[i])
            tokens = (lower_words[i][0], upper_words[i][0], all_upper_words[i][0])

            if i < len(lower_words) - 1:
                next_tokens = (lower_words[i+1][0], upper_words[i+1][0], all_upper_words[i+1][0])
            else:
                next_tokens = None

            # the first token of each word decides, whether the word is 1), 2) or 3)
            with torch.no_grad():
                logits = self.model(decoder_input_ids=current_ids, encoder_outputs=encoder_hidden_states).logits[0, -1]
                # correct token has to be in top 50
                top_k_50 = logits.topk(50).indices

            ranks = self.get_ranks(top_k_50, tokens)
            tokens_idx = ranks.index(min(ranks))
            tokens = torch.tensor(tokens_list[tokens_idx], device=device)

            # append complete word of 1), 2) or 3)
            current_ids = torch.cat([current_ids, tokens[None]], dim=-1)

            # Finally check for punctuation. For now, we assume that punctuation can only 
            # occur *after* a word, but not in the middle of the word.
            # Also punctuation can only always only take a single character.
            with torch.no_grad():
                logits = self.model(decoder_input_ids=current_ids, encoder_outputs=encoder_hidden_states).logits[0, -1]
                # punctuation has to be in top 10
                top_k_10 = logits.topk(10).indices

            next_token_ranks = self.get_ranks(top_k_10, next_tokens) if next_tokens is not None else [float("inf")]
            punc_ranks = self.get_ranks(top_k_10, self.punctuation)

            # punctuation is only added, if punctuation is more likely than next token **and** if 
            # punctuation is in top_10
            if not torch.tensor(punc_ranks, device=device).isinf().all() and min(punc_ranks) < min(next_token_ranks):
                punc_idx = punc_ranks.index(min(punc_ranks))
                punc = torch.tensor([self.punctuation[punc_idx]], device=device)
                current_ids = torch.cat([current_ids, punc[None]], dim=-1)

        orthographic_transcription = self.tokenizer.batch_decode(current_ids, skip_special_tokens=True)[0]
        # skip first character as it's always an empty space
        orthographic_transcription = orthographic_transcription[1:]

        return orthographic_transcription
