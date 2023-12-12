from typing import List, Optional, Union

import numpy as np
import requests
import torch
from pyannote.audio import Pipeline
from torchaudio import functional as F
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

from .utils.diarize_utils import match_segments


class ASRDiarizationPipeline:
    def __init__(
        self,
        asr_pipeline,
        diarization_pipeline,
    ):
        self.asr_pipeline = asr_pipeline
        self.sampling_rate = asr_pipeline.feature_extractor.sampling_rate

        self.diarization_pipeline = diarization_pipeline

    @classmethod
    def from_pretrained(
        cls,
        asr_model: Optional[str] = "openai/whisper-medium",
        *,
        diarizer_model: Optional[str] = "pyannote/speaker-diarization",
        chunk_length_s: Optional[int] = 30,
        use_auth_token: Optional[Union[str, bool]] = True,
        **kwargs,
    ):
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            chunk_length_s=chunk_length_s,
            token=use_auth_token,  # 08/25/2023: Changed argument from use_auth_token to token
            **kwargs,
        )
        diarization_pipeline = Pipeline.from_pretrained(diarizer_model, use_auth_token=use_auth_token)
        return cls(asr_pipeline, diarization_pipeline)

    def __call__(
        self,
        inputs: Union[np.ndarray, List[np.ndarray]],
        iou_threshold: float = 0.0,
        **kwargs,
    ) -> list[dict]:
        """
        Transcribe the audio sequence(s) given as inputs to text and label with speaker information. The input audio
        is first passed to the speaker diarization pipeline, which returns timestamps for 'who spoke when'. The audio
        is then passed to the ASR pipeline, which returns utterance-level transcriptions and their corresponding
        timestamps. The speaker diarizer timestamps are aligned with the ASR transcription timestamps to give
        speaker-labelled transcriptions. We perform a best intersection over union (IoU) to select the best match between
        the speaker diarizer segment and the ASR transcription segment.
        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                The inputs is either :
                    - `str` that is the filename of the audio file, the file will be read at the correct sampling rate
                      to get the waveform using *ffmpeg*. This requires *ffmpeg* to be installed on the system.
                    - `bytes` it is supposed to be the content of an audio file and is interpreted by *ffmpeg* in the
                      same way.
                    - (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`)
                        Raw audio at the correct sampling rate (no further check will be done)
                    - `dict` form can be used to pass raw audio sampled at arbitrary `sampling_rate` and let this
                      pipeline do the resampling. The dict must be in the format `{"sampling_rate": int, "raw":
                      np.array}` with optionally a `"stride": (left: int, right: int)` than can ask the pipeline to
                      treat the first `left` samples and last `right` samples to be ignored in decoding (but used at
                      inference to provide more context to the model). Only use `stride` with CTC models.
            iou_threshold (float):
                The threshold under which an IoU is considere too low.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update additional asr or diarization configuration parameters
                        - To update the asr configuration, use the prefix *asr_* for each configuration parameter.
                        - To update the diarization configuration, use the prefix *diarization_* for each configuration parameter.
                        - Added this support related to issue #25: 08/25/2023

        Return:
            A list of transcriptions. Each list item corresponds to one chunk / segment of transcription, and is a
            dictionary with the following keys:
                - **text** (`str` ) -- The recognized text.
                - **speaker** (`str`) -- The associated speaker.
                - **timestamps** (`tuple`) -- The start and end time for the chunk / segment.

        Note:
            If no match occur between the speaker diarizer segment and the ASR transcription segment, a `NO_SPEAKER` label
            will be assign as we can't infer properly the speaker of the segment.
        """
        kwargs_asr = {
            argument[len("asr_") :]: value for argument, value in kwargs.items() if argument.startswith("asr_")
        }

        kwargs_diarization = {
            argument[len("diarization_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("diarization_")
        }

        inputs, diarizer_inputs = self.preprocess(inputs)

        diarization = self.diarization_pipeline(
            {"waveform": diarizer_inputs, "sample_rate": self.sampling_rate},
            **kwargs_diarization,
        )

        dia_seg, dia_label = [], []
        for segment, _, label in diarization.itertracks(yield_label=True):
            dia_seg.append([segment.start, segment.end])
            dia_label.append(label)

        assert (
            dia_seg
        ), "The result from the diarization pipeline: `diarization_segments` is empty. No segments found from the diarization process."

        asr_out = self.asr_pipeline(
            {"array": inputs, "sampling_rate": self.sampling_rate},
            return_timestamps=True,
            **kwargs_asr,
        )
        segmented_preds = asr_out["chunks"]

        dia_seg = np.array(dia_seg)
        asr_seg = np.array([[*chunk["timestamp"]] for chunk in segmented_preds])

        asr_labels = match_segments(dia_seg, dia_label, asr_seg, threshold=iou_threshold, no_match_label="NO_SPEAKER")

        for i, label in enumerate(asr_labels):
            segmented_preds[i]["speaker"] = label

        return segmented_preds

    # Adapted from transformers.pipelines.automatic_speech_recognition.AutomaticSpeechRecognitionPipeline.preprocess
    # (see https://github.com/huggingface/transformers/blob/238449414f88d94ded35e80459bb6412d8ab42cf/src/transformers/pipelines/automatic_speech_recognition.py#L417)
    def preprocess(self, inputs):
        if isinstance(inputs, str):
            if inputs.startswith("http://") or inputs.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                inputs = requests.get(inputs).content
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.sampling_rate)

        if isinstance(inputs, dict):
            # Accepting `"array"` which is the key defined in `datasets` for better integration
            if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                raise ValueError(
                    "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                    '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array"
                )

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                # Remove path which will not be used from `datasets`.
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sampling_rate")
            inputs = _inputs
            if in_sampling_rate != self.sampling_rate:
                inputs = F.resample(torch.from_numpy(inputs), in_sampling_rate, self.sampling_rate).numpy()

        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for ASRDiarizePipeline")

        # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
        diarizer_inputs = torch.from_numpy(inputs).float()
        diarizer_inputs = diarizer_inputs.unsqueeze(0)

        return inputs, diarizer_inputs
