#!/usr/bin/env python3
import string
from typing import List, Union, Optional

import numpy as np
import torch
import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BeamSearchScorer


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

    def create_dicts(self, all_words: List[List[List[str]]]):
        all_words_flat = []
        max_length = 0
        end_of_words = torch.zeros((len(all_words), len(all_words[0])), dtype=torch.long)

        for i, words in enumerate(all_words):
            flat_words = [item for sublist in words for item in sublist]
            max_length = max(max_length, len(flat_words))
            all_words_flat.append(flat_words)

            end_of_words[i] = torch.tensor([len(w) for w in words], dtype=torch.long).cumsum(-1)

        token_dict = torch.zeros((len(all_words), max_length), dtype=torch.long) - 1
        for i, flat_words in enumerate(all_words_flat):
            token_dict[i][:len(flat_words)] = torch.tensor(flat_words)

        end_of_words = end_of_words.roll(1)
        end_of_words[:, 0] = 0

        return token_dict, end_of_words

    def __call__(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        transcript: Union[str, List[str]],
        sampling_rate: Optional[int] = None,
        num_beams: int = 3,
    ):
        device = self.model.device
        batch_size = 1
        vocab_size = self.model.config.vocab_size
        max_length = 50
        is_finished = False

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        beam_scorer = BeamSearchScorer(batch_size=batch_size, num_beams=num_beams, length_penalty=0.0, device=device)

        # there are three possibilities of how a fully orthographic version of `transcript` can
        # look like:
        # 1) All lower-cased, e.g. "hello"
        # 2) All upper-cased, e.g. "HELLO"
        # Option 2) is a rare case
        # 3) All lower-cased + space, e.g. " hello"
        # 4) The first letter upper-cased + space, e.g. " Hello"
        # 5) All upper-cased + space, e.g. " HELLO"
        # Option 5) is a rare case
        lower_words = self.convert_words([f"{word.lower()}" for word in transcript.split()])
        upper_words = self.convert_words([f"{word.lower().capitalize()}" for word in transcript.split()])

        _lower_words = self.convert_words([f" {word.lower()}" for word in transcript.split()])
        _upper_words = self.convert_words([f" {word.lower().capitalize()}" for word in transcript.split()])
        _all_upper_words = self.convert_words([f" {word.upper()}" for word in transcript.split()])

        all_words = [lower_words, upper_words, _lower_words, _upper_words, _all_upper_words]

        track_dict, track_ends = self.create_dicts(all_words)

        num_gen_tokens = torch.zeros((5, num_beams), dtype=torch.long)
        token_track = torch.zeros((num_beams,), dtype=torch.long)  # use -1 for no track between 1), ..., 5)

        # Quick quality check, lower-cased words have to be identical when decoding
        _flat_lower_words = [item for sublist in _lower_words for item in sublist]
        assert self.tokenizer.decode(_flat_lower_words)[1:] == transcript, f"Decoding of {transcript} is wrong."

        # Get encoder hidden states
        input_features = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt")["input_features"]
        input_features = input_features.to(device)

        with torch.no_grad():
            encoder_hidden_states = self.model.model.encoder(input_features).last_hidden_state
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_beams, dim=0)

        # Define `decoder_start_ids` as initial `current_ids`; they are quite specific to whisper-{}.en checkpoints
        decoder_start_ids = [self.model.config.decoder_start_token_id]
        if self.model.config.forced_decoder_ids is not None:
            for token_id in self.model.config.forced_decoder_ids:
                decoder_start_ids.append(token_id[-1])

        decoder_start_ids = torch.tensor(decoder_start_ids, dtype=torch.long, device=device)[None, :]
        decoder_start_ids = decoder_start_ids.repeat_interleave(num_beams, dim=0)
        current_ids = decoder_start_ids.broadcast_to(encoder_hidden_states.shape[:1] + decoder_start_ids.shape[1:])
        num_start_ids = decoder_start_ids.shape[-1]

        word_idx = torch.zeros_like(token_track)
        is_word_ended = torch.ones_like(token_track)

        # loop over words of possibilities 1), 2) and 3).
        while not is_finished:
            with torch.no_grad():
                logits = self.model(decoder_input_ids=current_ids, encoder_outputs=encoder_hidden_states).logits[:, -1]
                scores = torch.nn.functional.log_softmax(logits, dim=-1)

            num_puncs = torch.cat([current_ids == p for p in self.punctuation], dim=-1).sum(-1)
            steps = current_ids.shape[-1] - num_start_ids - num_puncs.cpu()

            # TODO(Patrick) process scores here!
            all_next_tokens = [None for _ in range(num_beams)]
            for i, is_ended in enumerate(is_word_ended):
                if is_ended:
                    start_steps = track_ends[:, word_idx[i]]
                    next_possible_tokens = torch.matmul(torch.nn.functional.one_hot(start_steps, track_dict.shape[-1]), track_dict.transpose(0, 1)).diagonal()
                    all_next_tokens[i] = next_possible_tokens

                    next_possible_tokens = torch.cat([torch.tensor(self.punctuation), next_possible_tokens], dim=-1)
                else:
                    next_possible_tokens = torch.tensor([track_dict[token_track[i], steps[i]]])
                    all_next_tokens[i] = next_possible_tokens

                next_possible_scores = scores[i][next_possible_tokens]
                scores[i] = torch.zeros_like(scores[i]) - float("inf")
                scores[i][next_possible_tokens] = next_possible_scores

            next_scores = scores + beam_scores[:, None].expand_as(scores)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                current_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id,
                beam_indices=None,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            current_ids = torch.cat([current_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            uses_punc = (torch.tensor(self.punctuation)[None, :].broadcast_to(num_beams, len(self.punctuation)) == beam_next_tokens.cpu()[:, None].broadcast_to(num_beams, len(self.punctuation))).any(-1)

            old_token_track = token_track.clone()
            for i, use_punc in enumerate(uses_punc):
                idx = beam_idx[i]
                if use_punc:
                    # if uses punctuation, we stay in track
                    token_track[i] = old_token_track[idx]
                elif not is_word_ended[idx]:
                    # if is in middle of the word, we stay in track
                    token_track[i] = old_token_track[idx]
                else:
                    # if it's a new word, we have to find out which track was chosen
                    token_track[i] = (all_next_tokens[idx] == beam_next_tokens[i].cpu()).nonzero().item()

            old_word_idx = word_idx.clone()
            for i, ids in enumerate(current_ids[:, num_start_ids:]):
                idx = beam_idx[i]
                potential_word = all_words[token_track[i]][old_word_idx[i]]
                if ids.shape[0] >= len(potential_word) and ids.cpu()[-len(potential_word):].tolist() == potential_word:
                    word_idx[i] = old_word_idx[idx] + 1
                    is_word_ended[i] = 1
                elif uses_punc[i]:
                    word_idx[i] = old_word_idx[idx]
                    is_word_ended[i] = 1
                else:
                    word_idx[i] = old_word_idx[idx]
                    is_word_ended[i] = 0

            print("Out:", self.tokenizer.batch_decode(current_ids))

        sequence_outputs = beam_scorer.finalize(
            current_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            max_length=max_length,
            beam_indices=None,
        )

        orthographic_transcription = self.tokenizer.batch_decode(current_ids, skip_special_tokens=True)[0]
        # skip first character as it's always an empty space
        orthographic_transcription = orthographic_transcription[1:]

        print(orthographic_transcription)

        return orthographic_transcription

                # correct token has to be in top 50
#                top_k_50 = logits.topk(50).indices
#
#            ranks = self.get_ranks(top_k_50, tokens)
#            tokens_idx = ranks.index(min(ranks))
#            tokens = torch.tensor(tokens_list[tokens_idx], device=device)
#
            # append complete word of 1), 2) or 3)
#            current_ids = torch.cat([current_ids, tokens[None]], dim=-1)
#
            # Finally check for punctuation. For now, we assume that punctuation can only
            # occur *after* a word, but not in the middle of the word.
            # Also punctuation can only always only take a single character.
#            with torch.no_grad():
#                logits = self.model(decoder_input_ids=current_ids, encoder_outputs=encoder_hidden_states).logits[0, -1]
                # punctuation has to be in top 10
#                top_k_10 = logits.topk(10).indices
#
#            next_token_ranks = self.get_ranks(top_k_10, next_tokens) if next_tokens is not None else [float("inf")]
#            punc_ranks = self.get_ranks(top_k_10, self.punctuation)
#
            # punctuation is only added, if punctuation is more likely than next token **and** if 
            # punctuation is in top_10
#            if not torch.tensor(punc_ranks, device=device).isinf().all() and min(punc_ranks) < min(next_token_ranks):
#                punc_idx = punc_ranks.index(min(punc_ranks))
#                punc = torch.tensor([self.punctuation[punc_idx]], device=device)
#                current_ids = torch.cat([current_ids, punc[None]], dim=-1)
#
#        orthographic_transcription = self.tokenizer.batch_decode(current_ids, skip_special_tokens=True)[0]
        # skip first character as it's always an empty space
#        orthographic_transcription = orthographic_transcription[1:]
#
#        return orthographic_transcription
