#!/usr/bin/env python3
import string
from typing import List, Optional, Union

import numpy as np
import torch
from scipy.signal import resample
from transformers import (BeamSearchScorer, WhisperForConditionalGeneration,
                          WhisperProcessor)


class PunctuationRestorer:
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
        for i in range(len(self.tokenizer)):
            if self.tokenizer.convert_ids_to_tokens(i) in list(string.punctuation):
                punctuation_tokens.append(i)
        return punctuation_tokens

    def convert_words(self, words):
        word_tokens = []
        for word in words:
            tokens = self.tokenizer.tokenize(word)
            word_tokens.append(self.tokenizer.convert_tokens_to_ids(tokens))

        return word_tokens

    def __call__(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        transcript: Union[str, List[str]],
        sampling_rate: Optional[int] = None,
        num_beams: int = 1,
    ):
        # 1. Define some import variables
        device = self.model.device
        batch_size = 1
        vocab_size = self.model.config.vocab_size
        num_words = len(transcript.split())

        # 2. Define all possible words that can be generated
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

        # put all possible words in a "all_words" list
        all_words = [lower_words, upper_words, _lower_words, _upper_words, _all_upper_words]

        # 3. Sanity check
        # Quick quality check, lower-cased words have to be identical when decoding
        _flat_lower_words = [item for sublist in _lower_words for item in sublist]
        assert (
            self.tokenizer.decode(_flat_lower_words)[1:] == transcript.lower()
        ), f"Decoding of {transcript} is wrong."

        # Resample audio if necessary
        if sampling_rate != self.model_sampling_rate:
            audio = resample(audio, int(audio.shape[-1] * self.model_sampling_rate / sampling_rate))

        # 4. Get encoder hidden states
        input_features = self.processor(audio, sampling_rate=self.model_sampling_rate, return_tensors="pt")[
            "input_features"
        ]
        input_features = input_features.to(device)

        with torch.no_grad():
            encoder_hidden_states = self.model.model.encoder(input_features).last_hidden_state
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_beams, dim=0)

        # 5. Prepare initial ids to be decoded
        # Define `decoder_start_ids` as initial `current_ids`; they are quite specific to whisper-{}.en checkpoints
        decoder_start_ids = [self.model.config.decoder_start_token_id]
        if self.model.config.forced_decoder_ids is not None:
            for token_id in self.model.config.forced_decoder_ids:
                decoder_start_ids.append(token_id[-1])

        decoder_start_ids = torch.tensor(decoder_start_ids, dtype=torch.long, device=device)[None, :]
        decoder_start_ids = decoder_start_ids.repeat_interleave(num_beams, dim=0)
        current_ids = decoder_start_ids.broadcast_to(encoder_hidden_states.shape[:1] + decoder_start_ids.shape[1:])

        # 6. Define the beam scorer
        if num_beams > 1:
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
            beam_scores[:, 1:] = -1e9
            beam_scores = beam_scores.view((batch_size * num_beams,))
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size, num_beams=num_beams, length_penalty=0.0, device=device
            )
        else:
            beam_scores = torch.zeros((batch_size,), dtype=torch.float, device=device)
            beam_scorer = None

        # 7. Create some running variables that will be changed during the decoding
        num_start_ids = decoder_start_ids.shape[-1]
        token_track = [[0] for _ in range(num_beams)]
        word_idx = torch.zeros((num_beams,), dtype=torch.long)
        is_word_ended = torch.ones_like(word_idx)
        in_word_index = torch.zeros_like(word_idx)
        uses_punc = torch.zeros_like(word_idx)

        # 8. Start the decoding loop
        while True:
            # 8.1 forward the current ideas and retrieve log softmax
            with torch.no_grad():
                encoder_outputs = (encoder_hidden_states,)
                logits = self.model(decoder_input_ids=current_ids, encoder_outputs=encoder_outputs).logits[:, -1]
                scores = torch.nn.functional.log_softmax(logits, dim=-1)

            # 8.2 Constrain tokens that can be generated to tokens of words as defined above in `all_words`
            all_next_tokens = [None for _ in range(num_beams)]
            for i, is_ended in enumerate(is_word_ended):
                can_finish = word_idx[i] >= num_words

                if can_finish and not uses_punc[i]:
                    # if word is completed, we allow it to be finished or add punctuation and then finish
                    next_possible_tokens = torch.tensor(self.punctuation + [self.model.config.eos_token_id])
                elif can_finish:
                    # if word is completed and it has already seen punctuation, we force it to be finished
                    next_possible_tokens = torch.tensor([self.model.config.eos_token_id])
                elif is_ended:
                    # if a word has ended, all next words can be begin of new word
                    next_possible_tokens = torch.tensor([w[word_idx[i]][0] for w in all_words])
                    all_next_tokens[i] = next_possible_tokens

                    # if previously punctuation wasn't used, it can be used now
                    if not uses_punc[i]:
                        next_possible_tokens = torch.cat(
                            [torch.tensor(self.punctuation), next_possible_tokens], dim=-1
                        )
                else:
                    # here we are in the scenario that we're inside one or more words already. Then we simply need to continue this word
                    in_word_step = in_word_index[i]
                    next_possible_tokens = []
                    for track in token_track[i]:
                        tokens_in_word = all_words[track][word_idx[i]]
                        next_possible_tokens.append(tokens_in_word[in_word_step])

                    next_possible_tokens = torch.tensor(next_possible_tokens, dtype=torch.long)
                    all_next_tokens[i] = next_possible_tokens

                # constrain scores
                next_possible_scores = scores[i][next_possible_tokens].clone()
                scores[i] = torch.zeros_like(scores[i]) - float("inf")
                scores[i][next_possible_tokens] = next_possible_scores

            # 8.3 Run beam search
            next_scores = scores + beam_scores[:, None].expand_as(scores)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            if beam_scorer is not None:
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
            else:
                beam_scores = next_token_scores[0, :1]
                beam_next_tokens = next_tokens[0, :1]
                beam_idx = next_indices[0, :1]

            # 8.4 Concatenate predicted ids to current ids
            current_ids = torch.cat([current_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            # 8.5 Break when greedy search generates EOS
            # if we do greedy search we can already finish here
            if num_beams == 1 and current_ids[0, -1].item() == self.model.config.eos_token_id:
                break

            # 8.6 Check if punctuation was generated
            uses_punc = (
                torch.tensor(self.punctuation)[None, :].broadcast_to(num_beams, len(self.punctuation))
                == beam_next_tokens.cpu()[:, None].broadcast_to(num_beams, len(self.punctuation))
            ).any(-1)

            # 8.7 Determine which token track was chosen
            old_token_track = token_track.copy()
            for i, use_punc in enumerate(uses_punc):
                idx = beam_idx[i]
                if use_punc:
                    # if uses punctuation, we stay in track
                    token_track[i] = old_token_track[idx]
                elif not is_word_ended[idx]:
                    if len(old_token_track[idx]) > 1:
                        selected_index = (all_next_tokens[idx] == beam_next_tokens[i].cpu()).nonzero().flatten().item()
                    else:
                        selected_index = 0

                    token_track[i] = [old_token_track[idx][selected_index]]
                else:
                    # if it's a new word, we have to find out which track(s) was chosen
                    token_track[i] = (all_next_tokens[idx] == beam_next_tokens[i].cpu()).nonzero().flatten().tolist()

            # 8.8 Determine whether we are at end of word
            old_word_idx = word_idx.clone()
            old_is_word_index = in_word_index.clone()
            for i, ids in enumerate(current_ids[:, num_start_ids:]):
                idx = beam_idx[i]

                # Don't bother if word is already finished
                if old_word_idx[idx] >= num_words:
                    word_idx[i] = old_word_idx[idx]
                    continue

                potential_word = all_words[token_track[i][0]][old_word_idx[idx]]
                has_ended = (
                    ids.shape[0] >= len(potential_word)
                    and ids.cpu()[-len(potential_word) :].tolist() == potential_word
                )

                if has_ended:
                    word_idx[i] = old_word_idx[idx] + 1
                    is_word_ended[i] = 1
                    in_word_index[i] = 0
                elif uses_punc[i]:
                    word_idx[i] = old_word_idx[idx]
                    is_word_ended[i] = 1
                    in_word_index[i] = 0
                else:
                    word_idx[i] = old_word_idx[idx]
                    is_word_ended[i] = 0
                    in_word_index[i] = old_is_word_index[idx] + 1

            # 8.9 If we do beam search we only finish if all beams have been run
            if num_beams > 1 and len(beam_scorer._beam_hyps[0]) >= num_beams:
                break

        # 9. Finalize beams and retrieve final sequences as well as log probs
        if beam_scorer is not None:
            # max-lenth doesn't matter here
            sequence_outputs = beam_scorer.finalize(
                current_ids,
                beam_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id,
                max_length=1000,
                beam_indices=None,
            )
            final_sequences = sequence_outputs["sequences"].cpu()
            final_probs = sequence_outputs["sequence_scores"].cpu().item()
        else:
            final_sequences = current_ids.cpu()
            final_probs = beam_scores.cpu().item()

        # 10. Decode generated tokens
        orthographic_transcription = self.tokenizer.batch_decode(final_sequences, skip_special_tokens=True)[0]
        # skip first character as it's always an empty space
        orthographic_transcription = orthographic_transcription[1:]

        return orthographic_transcription, final_probs

    @property
    def model_sampling_rate(self):
        return self.processor.feature_extractor.sampling_rate
