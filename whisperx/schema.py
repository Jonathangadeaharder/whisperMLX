from collections.abc import Callable
from typing import TypedDict

# NotRequired landed in typing on 3.11; the project supports 3.10, so use
# typing_extensions (a transitive dep of torch/transformers) unconditionally.
from typing_extensions import NotRequired

ProgressCallback = Callable[[float], None] | None


class SingleWordSegment(TypedDict):
    """
    A single word of a speech.
    """

    word: str
    start: float
    end: float
    score: float
    speaker: NotRequired[str | None]


class SingleCharSegment(TypedDict):
    """
    A single char of a speech.
    """

    char: str
    start: float
    end: float
    score: float


class SingleSegment(TypedDict):
    """
    A single segment (up to multiple sentences) of a speech.
    """

    start: float
    end: float
    text: str
    avg_logprob: NotRequired[float | None]
    speaker: NotRequired[str | None]


class SegmentData(TypedDict):
    """
    Temporary processing data used during alignment.
    Contains cleaned and preprocessed data for each segment.
    """

    clean_char: list[str]  # Cleaned characters that exist in model dictionary
    clean_cdx: list[int]  # Original indices of cleaned characters
    clean_wdx: list[int]  # Indices of words containing valid characters
    sentence_spans: list[tuple[int, int]]  # Start and end indices of sentences


class SingleAlignedSegment(TypedDict):
    """
    A single segment (up to multiple sentences) of a speech with word alignment.
    """

    start: float
    end: float
    text: str
    avg_logprob: NotRequired[float | None]
    words: NotRequired[list[SingleWordSegment]]
    chars: list[SingleCharSegment] | None
    speaker: NotRequired[str | None]


class TranscriptionResult(TypedDict):
    """
    A list of segments and word segments of a speech.
    """

    segments: list[SingleSegment]
    language: str
    speaker_embeddings: NotRequired[dict[str, list[float]] | None]


class AlignedTranscriptionResult(TypedDict):
    """
    A list of segments and word segments of a speech.
    """

    segments: list[SingleAlignedSegment]
    word_segments: list[SingleWordSegment]
    language: NotRequired[str]
    speaker_embeddings: NotRequired[dict[str, list[float]] | None]
