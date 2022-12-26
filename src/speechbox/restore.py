#!/usr/bin/env python3
import string
from typing import List, Union

import numpy as np
import torch
import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class Restorer:
    def __init__(self, model: WhisperForConditionalGeneration, processor: WhisperProcessor):
        self.processor = processor
        self.model = model
        self.punctuation = self.get_punctuation_tokens()

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path):
        model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_or_path, low_cpu_mem_usage=True)
        processor = WhisperProcessor.from_pretrained(pretrained_model_or_path)
        return cls(model, processor)

    def get_punctuation_tokens(self):
        punctuation_tokens = []
        for i in tqdm.tqdm(range(len(self.processor.tokenizer))):
            if self.processor.tokenizer.convert_ids_to_tokens(i) in list(string.punctuation):
                punctuation_tokens.append(i)
        return punctuation_tokens

    def __call__(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        transcript: Union[str, List[str]],
    ):
        # whisper always starts as follows
        lower_words = [f"Ġ{word}" for word in transcript.split()]
        target_lower_words = [
            self.processor.tokenizer(word, add_special_tokens=False).input_ids for word in lower_words
        ]

        upper_words = [f"Ġ{word.capitalize()}" for word in transcript.split()]
        target_upper_words = [
            self.processor.tokenizer(word, add_special_tokens=False).input_ids for word in upper_words
        ]

        all_upper_words = [f"Ġ{word.upper()}" for word in transcript.split()]
        target_all_upper_words = [
            self.processor.tokenizer(word, add_special_tokens=False).input_ids for word in all_upper_words
        ]

        input_features = self.processor(audio, return_tensors="pt")["input_features"]
        with torch.no_grad():
            encoder_hidden_states = self.model.model.encoder(input_features).last_hidden_state

        decoder_start_tokens = [50257, 50362]

        # start tokens for English
        decoder_start_ids = torch.tensor(decoder_start_tokens, dtype=torch.long)[None, :]
        current_ids = decoder_start_ids.broadcast_to(encoder_hidden_states.shape[:1] + decoder_start_ids.shape[1:])

        for l_w, u_w, up_w in zip(target_lower_words, target_upper_words, target_all_upper_words):
            is_finished = False
            while not is_finished:
                logits = self.model(decoder_input_ids=current_ids, encoder_outputs=encoder_hidden_states).logits[:, -1]
                top_k_10 = logits.topk(10).indices
                import ipdb

                ipdb.set_trace()
