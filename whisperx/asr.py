import os
from typing import List, Optional, Union

import mlx_whisper
import numpy as np
import torch
from transformers import WhisperTokenizer

from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio
from whisperx.schema import SingleSegment, TranscriptionResult, ProgressCallback
from whisperx.vads import Vad, Silero, Pyannote
from whisperx.log_utils import get_logger

logger = get_logger(__name__)

# Map whisperX model aliases to mlx-community Hugging Face repos.
# Pass through if the arch already names a repo (contains "/") or an
# mlx-community prefix.
_MLX_COMMUNITY = "mlx-community/whisper-"


def _resolve_model_path(whisper_arch: str) -> str:
    """Resolve a whisperX model alias to an mlx-whisper model path/repo."""
    if "/" in whisper_arch:
        return whisper_arch
    if whisper_arch.startswith("mlx"):
        return whisper_arch
    return _MLX_COMMUNITY + whisper_arch


def find_numeral_symbol_tokens(tokenizer) -> List[int]:
    """Token ids whose decoded string contains a numeral or currency symbol."""
    numeral_symbol_tokens: List[int] = []
    eot = getattr(tokenizer, "eot", None)
    if eot is None:
        eot = tokenizer.vocab_size
    for i in range(eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens


class MlxWhisperPipeline:
    """Pipeline that runs VAD then transcribes each segment with mlx-whisper.

    Replaces whisperX's FasterWhisperPipeline. Per-segment transcription only
    (no batched encoder runs); mlx-whisper runs on the Apple Silicon GPU.
    """

    def __init__(
        self,
        model_path: str,
        vad,
        vad_params: dict,
        mlx_options: dict,
        language: Optional[str] = None,
        suppress_numerals: bool = False,
        suppress_tokens: Optional[List[int]] = None,
    ):
        self.model_path = model_path
        self.vad_model = vad
        self._vad_params = vad_params
        self._mlx_options = mlx_options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self.suppress_tokens = suppress_tokens

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size: int = 30,
        print_progress: bool = False,
        combined_progress: bool = False,
        verbose: bool = False,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        if batch_size not in (None, 0, 1):
            logger.info(
                "batch_size=%s ignored; mlx-whisper transcribes one segment at a time",
                batch_size,
            )
        if num_workers:
            logger.info("num_workers=%s ignored by mlx-whisper pipeline", num_workers)

        if isinstance(audio, str):
            audio = load_audio(audio)

        # Pre-process audio and merge chunks as defined by the VAD child class.
        # Manually assigned VAD models follow the pyannote toolkit interface.
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks = self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks

        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )

        language = language or self.preset_language
        task = task or "transcribe"

        segments: List[SingleSegment] = []
        total_segments = len(vad_segments)
        for idx, seg in enumerate(vad_segments):
            f1 = int(seg["start"] * SAMPLE_RATE)
            f2 = int(seg["end"] * SAMPLE_RATE)
            audio_slice = audio[f1:f2]

            seg_kwargs = dict(self._mlx_options)
            if language is not None:
                seg_kwargs["language"] = language
            seg_kwargs["task"] = task
            if self.suppress_tokens is not None:
                seg_kwargs["suppress_tokens"] = self.suppress_tokens

            result = mlx_whisper.transcribe(
                audio_slice,
                path_or_hf_repo=self.model_path,
                **seg_kwargs,
            )
            if language is None:
                language = result.get("language")
                if language is not None and self.preset_language is None:
                    logger.info("Detected language: %s", language)

            seg_offset = seg["start"]
            for sub in result.get("segments", []):
                text = sub.get("text", "").strip()
                if not text:
                    continue
                segments.append(
                    {
                        "text": text,
                        "start": round(seg_offset + sub.get("start", 0.0), 3),
                        "end": round(seg_offset + sub.get("end", 0.0), 3),
                        "avg_logprob": sub.get("avg_logprob"),
                    }
                )
                if verbose:
                    print(
                        f"Transcript: [{segments[-1]['start']} --> "
                        f"{segments[-1]['end']}] {text}"
                    )

            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = (
                    base_progress / 2 if combined_progress else base_progress
                )
                print(f"Progress: {percent_complete:.2f}%...")
            if progress_callback is not None:
                progress_callback(((idx + 1) / total_segments) * 100)

        return {"segments": segments, "language": language or "en"}


def load_model(
    whisper_arch: str,
    device: str,
    device_index: int = 0,
    compute_type: str = "default",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad] = None,
    vad_method: Optional[str] = "pyannote",
    vad_options: Optional[dict] = None,
    model: Optional[object] = None,
    task: str = "transcribe",
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    threads: int = 4,
    use_auth_token: Optional[Union[str, bool]] = None,
) -> MlxWhisperPipeline:
    """Load a Whisper model for inference via mlx-whisper.

    Args:
        whisper_arch: Whisper model alias (tiny/base/small/medium/large/large-v2/
            large-v3/large-v3-turbo/turbo) or an mlx-community repo / local path.
        device: ignored. mlx-whisper always runs on the Apple Silicon GPU.
        device_index: ignored.
        compute_type: ignored. mlx-whisper uses the model's quantization as-is.
        asr_options: dict overriding default ASR options (see below).
        language: spoken language; None enables per-auto-detection.
        vad_model: manually assigned VAD (overrides vad_method).
        vad_method: "pyannote" or "silero".
        vad_options: dict overriding VAD defaults (chunk_size, vad_onset, vad_offset).
        model: ignored (kept for API compatibility).
        task: "transcribe" or "translate".
        download_root: ignored (mlx-whisper uses the HF cache).
        local_files_only: if True, mlx-whisper is restricted to cached models.
        threads: ignored by mlx-whisper.
        use_auth_token: Hugging Face token for gated mlx-community models.

    Returns:
        An MlxWhisperPipeline.
    """
    if device == "cuda":
        logger.warning(
            "device='cuda' requested but mlx-whisper runs on Apple Silicon GPU; ignoring"
        )
    if compute_type != "default":
        logger.info(
            "compute_type=%s ignored; mlx-whisper uses the model's quantization",
            compute_type,
        )

    if whisper_arch.endswith(".en"):
        language = "en"

    model_path = _resolve_model_path(whisper_arch)

    default_asr_options = {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "multilingual": True,
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
        "hotwords": None,
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options.pop("suppress_numerals", False)
    default_asr_options.pop("multilingual", None)

    # mlx-whisper has no beam search decoder: beam_size, patience, best_of,
    # length_penalty are dropped. Temperature tuple kept for fallback.
    temperatures = default_asr_options.get("temperatures", [0.0])
    temperature = tuple(temperatures) if len(temperatures) > 1 else temperatures[0]

    mlx_options = {
        "temperature": temperature,
        "compression_ratio_threshold": default_asr_options.get(
            "compression_ratio_threshold"
        ),
        "logprob_threshold": default_asr_options.get("log_prob_threshold"),
        "no_speech_threshold": default_asr_options.get("no_speech_threshold"),
        "condition_on_previous_text": default_asr_options.get(
            "condition_on_previous_text"
        ),
        "initial_prompt": default_asr_options.get("initial_prompt"),
        "word_timestamps": default_asr_options.get("word_timestamps"),
        "prepend_punctuations": default_asr_options.get("prepend_punctuations"),
        "append_punctuations": default_asr_options.get("append_punctuations"),
        "clip_timestamps": default_asr_options.get("clip_timestamps") or "0",
        "hallucination_silence_threshold": default_asr_options.get(
            "hallucination_silence_threshold"
        ),
        "verbose": False,
    }
    # Drop None-valued options so mlx-whisper applies its own defaults.
    mlx_options = {k: v for k, v in mlx_options.items() if v is not None}

    suppress_tokens = default_asr_options.get("suppress_tokens", [-1])
    if suppress_numerals:
        tokenizer = WhisperTokenizer.from_pretrained(model_path)
        numeral_symbol_tokens = find_numeral_symbol_tokens(tokenizer)
        logger.info("Suppressing numeral and symbol tokens")
        suppress_tokens = list(set(numeral_symbol_tokens + suppress_tokens))

    default_vad_options = {
        "chunk_size": 30,
        "vad_onset": 0.500,
        "vad_offset": 0.363,
    }
    if vad_options is not None:
        default_vad_options.update(vad_options)

    # Manually assigned vad_model has higher priority than vad_method.
    if vad_model is not None:
        logger.info("Using manually assigned vad_model; vad_method ignored")
    elif vad_method == "silero":
        vad_model = Silero(**default_vad_options)
    elif vad_method == "pyannote":
        # mlx-whisper ignores device; pyannote VAD runs on torch (CPU on Apple
        # Silicon for now).
        device_vad = "cpu"
        vad_model = Pyannote(
            torch.device(device_vad),
            token=use_auth_token,
            **default_vad_options,
        )
    else:
        raise ValueError(f"Invalid vad_method: {vad_method}")

    return MlxWhisperPipeline(
        model_path=model_path,
        vad=vad_model,
        vad_params=default_vad_options,
        mlx_options=mlx_options,
        language=language,
        suppress_numerals=suppress_numerals,
        suppress_tokens=suppress_tokens,
    )
