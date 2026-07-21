import argparse
import gc
import os
import warnings

import numpy as np
import pandas as pd

from whisperx.alignment import align, load_align_model
from whisperx.asr import load_model
from whisperx.audio import load_audio
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.log_utils import get_logger
from whisperx.schema import AlignedTranscriptionResult, TranscriptionResult
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE, get_writer

logger = get_logger(__name__)


def transcribe_task(args: dict, parser: argparse.ArgumentParser):  # noqa: PLR0912, PLR0915
    """Transcription task to be called from CLI.

    Args:
        args: Dictionary of command-line arguments.
        parser: argparse.ArgumentParser object.
    """
    # fmt: off

    model_name: str = args.pop("model")
    batch_size: int = args.pop("batch_size")
    model_dir: str = args.pop("model_dir")
    model_cache_only: bool = args.pop("model_cache_only")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    device_index: int = args.pop("device_index")
    compute_type: str = args.pop("compute_type")
    verbose: bool = args.pop("verbose")

    # model_flush: bool = args.pop("model_flush")
    os.makedirs(output_dir, exist_ok=True)

    align_model: str = args.pop("align_model")
    interpolate_method: str = args.pop("interpolate_method")
    no_align: bool = args.pop("no_align")
    task: str = args.pop("task")
    if task == "translate":
        # translation cannot be aligned
        no_align = True

    return_char_alignments: bool = args.pop("return_char_alignments")

    hf_token: str = args.pop("hf_token")
    vad_method: str = args.pop("vad_method")
    vad_onset: float = args.pop("vad_onset")
    vad_offset: float = args.pop("vad_offset")

    chunk_size: int = args.pop("chunk_size")

    diarize: bool = args.pop("diarize")
    min_speakers: int = args.pop("min_speakers")
    max_speakers: int = args.pop("max_speakers")
    diarize_model_name: str = args.pop("diarize_model")
    print_progress: bool = args.pop("print_progress")
    return_speaker_embeddings: bool = args.pop("speaker_embeddings")

    if return_speaker_embeddings and not diarize:
        warnings.warn("--speaker_embeddings has no effect without --diarize", stacklevel=2)

    if args["language"] is not None:
        args["language"] = args["language"].lower()
        if args["language"] not in LANGUAGES:
            if args["language"] in TO_LANGUAGE_CODE:
                args["language"] = TO_LANGUAGE_CODE[args["language"]]
            else:
                raise ValueError(f"Unsupported language: {args['language']}")

    if model_name.endswith(".en") and args["language"] != "en":
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but received '{args['language']}'; using English instead.", stacklevel=2
            )
        args["language"] = "en"
    align_language = (
        args["language"] if args["language"] is not None else "en"
    )  # default to loading english if not specified

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    # --threads is accepted for CLI compatibility but mlx-whisper ignores it.
    args.pop("threads")

    asr_options = {
        "beam_size": args.pop("beam_size"),
        "patience": args.pop("patience"),
        "length_penalty": args.pop("length_penalty"),
        "temperatures": temperature,
        "compression_ratio_threshold": args.pop("compression_ratio_threshold"),
        "log_prob_threshold": args.pop("logprob_threshold"),
        "no_speech_threshold": args.pop("no_speech_threshold"),
        "condition_on_previous_text": False,
        "initial_prompt": args.pop("initial_prompt"),
        "hotwords": args.pop("hotwords"),
        "suppress_tokens": [int(x) for x in args.pop("suppress_tokens").split(",")],
        "suppress_numerals": args.pop("suppress_numerals"),
    }

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if no_align:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} not possible with --no_align")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width", stacklevel=2)
    writer_args = {arg: args.pop(arg) for arg in word_options}

    # Part 1: VAD & ASR Loop
    # The list holds either TranscriptionResult (pre-align) or
    # AlignedTranscriptionResult (post-align); the writer accepts both.
    results: list[tuple[TranscriptionResult | AlignedTranscriptionResult, str]] = []
    # model = load_model(model_name, device=device, download_root=model_dir)
    model = load_model(
        model_name,
        device=device,
        device_index=device_index,
        download_root=model_dir,
        compute_type=compute_type,
        language=args["language"],
        asr_options=asr_options,
        vad_method=vad_method,
        vad_options={
            "chunk_size": chunk_size,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        },
        task=task,
        local_files_only=model_cache_only,
        use_auth_token=hf_token,
    )

    for audio_path in args.pop("audio"):
        audio = load_audio(audio_path)
        # >> VAD & ASR
        logger.info("Performing transcription...")
        result = model.transcribe(
            audio,
            batch_size=batch_size,
            chunk_size=chunk_size,
            print_progress=print_progress,
            verbose=verbose,
        )
        results.append((result, audio_path))

    # Unload Whisper and VAD
    del model
    gc.collect()

    # Part 2: Align Loop
    if not no_align:
        tmp_results = results
        results = []
        align_model, align_metadata = load_align_model(
            align_language, device, model_name=align_model, model_dir=model_dir, model_cache_only=model_cache_only
        )
        for align_in, audio_path in tmp_results:
            # >> Align
            # lazily load audio from part 1 when there is a single result
            input_audio = audio_path if len(tmp_results) > 1 else audio

            result = align_in
            if align_model is not None and len(result["segments"]) > 0:
                if result.get("language", "en") != align_metadata["language"]:
                    # load new language
                    logger.info(
                        f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language..."
                    )
                    align_model, align_metadata = load_align_model(
                        result["language"], device, model_dir=model_dir, model_cache_only=model_cache_only
                    )
                logger.info("Performing alignment...")
                # result widens from TranscriptionResult to the aligned shape;
                # the writer accepts both. pyrefly keeps the first declared
                # type, so suppress the bad-assignment on this rebind.
                result = align(  # pyrefly: ignore[bad-assignment]
                    result["segments"],
                    align_model,
                    align_metadata,
                    input_audio,
                    device,
                    interpolate_method=interpolate_method,
                    return_char_alignments=return_char_alignments,
                    print_progress=print_progress,
                )

            results.append((result, audio_path))  # pyrefly: ignore[bad-argument-type]

        # Unload align model
        del align_model
        gc.collect()

    # >> Diarize
    if diarize:
        if hf_token is None:
            logger.warning(
                "No --hf_token provided, needs to be saved in environment variable, otherwise will throw error loading diarization model"
            )
        tmp_results = results
        logger.info("Performing diarization...")
        logger.info(f"Using model: {diarize_model_name}")
        results = []
        diarize_model = DiarizationPipeline(model_name=diarize_model_name, token=hf_token, device=device, cache_dir=model_dir)
        for res, input_audio_path in tmp_results:
            diarize_result = diarize_model(
                input_audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=return_speaker_embeddings
            )

            diarize_segments: pd.DataFrame
            speaker_embeddings: dict[str, list[float]] | None
            if return_speaker_embeddings:
                # pyrefly can't narrow the tuple branch of the union return.
                diarize_segments, speaker_embeddings = diarize_result  # pyrefly: ignore[bad-assignment]
            else:
                diarize_segments = diarize_result  # pyrefly: ignore[bad-assignment]
                speaker_embeddings = None

            res = assign_word_speakers(diarize_segments, res, speaker_embeddings)  # noqa: PLW2901 - intentional rebind to enriched result
            results.append((res, input_audio_path))
    # >> Write
    for result, audio_path in results:
        result["language"] = align_language
        # writer expects dict; TranscriptionResult is a TypedDict (dict at runtime).
        writer(result, audio_path, writer_args)  # pyrefly: ignore[bad-argument-type]
