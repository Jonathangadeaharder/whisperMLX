import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.log_utils import get_logger
from whisperx.schema import (
    AlignedTranscriptionResult,
    ProgressCallback,
    SingleWordSegment,
    TranscriptionResult,
)

logger = get_logger(__name__)

# Embedding extraction windows for diarization clustering.
EMB_WINDOW = 3.0
EMB_STEP = 1.0


class IntervalTree:
    """Interval tree for fast overlap queries via sorted array + binary search.

    O(n) space, O(log n) query time instead of O(n) linear scan.
    """

    def __init__(self, intervals: list[tuple[float, float, str]]):
        if not intervals:
            self.starts = np.array([])
            self.ends = np.array([])
            self.speakers: list[str] = []
            return
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        self.starts = np.array([i[0] for i in sorted_intervals], dtype=np.float64)
        self.ends = np.array([i[1] for i in sorted_intervals], dtype=np.float64)
        self.speakers = [i[2] for i in sorted_intervals]

    def query(self, start: float, end: float) -> list[tuple[str, float]]:
        if len(self.starts) == 0:
            return []
        right_idx = np.searchsorted(self.starts, end, side="left")
        if right_idx == 0:
            return []
        candidates = slice(0, right_idx)
        overlaps = (self.starts[candidates] < end) & (self.ends[candidates] > start)
        results = []
        for idx in np.nonzero(overlaps)[0]:
            intersection = min(self.ends[idx], end) - max(self.starts[idx], start)
            if intersection > 0:
                results.append((self.speakers[idx], intersection))
        return results

    def find_nearest(self, time: float) -> str | None:
        if len(self.starts) == 0:
            return None
        mids = (self.starts + self.ends) / 2
        nearest_idx = np.argmin(np.abs(mids - time))
        return self.speakers[nearest_idx]


class DiarizationPipeline:
    """MLX diarization: segmentation + WeSpeaker embeddings + clustering.

    Replaces pyannote.audio.Pipeline. Uses the MLX pyannote segmentation
    (phase 4) for speech regions and MLX WeSpeaker ResNet34 for 256-dim
    speaker embeddings, clustered via sklearn AgglomerativeClustering.
    """

    def __init__(
        self,
        model_name=None,  # noqa: ARG002
        token=None,  # noqa: ARG002
        device=None,  # noqa: ARG002
        cache_dir=None,  # noqa: ARG002
    ):
        # Lazy imports: mlx/weights load only when diarization is used.
        from whisperx.mlx_models.pyannote_segmentation import segment_audio  # noqa: PLC0415
        from whisperx.mlx_models.wespeaker import _load_weights as load_ws  # noqa: PLC0415
        from whisperx.mlx_models.wespeaker import embed  # noqa: PLC0415

        logger.info("Loading MLX diarization pipeline (segmentation + WeSpeaker)...")
        self._segment_audio = segment_audio
        self._embed = embed
        self._wespeaker_weights = load_ws()
        # Segmentation produces speech prob per frame; binarize with onset/offset.
        self.vad_onset = 0.5
        self.vad_offset = 0.363

    def _binarize_segments(self, scores, frame_times):
        """Hysteresis thresholding -> speech segments."""
        # Lazy import: VAD binarize helper pulls in numpy-only utilities.
        from whisperx.vads.pyannote import _Binarize  # noqa: PLC0415

        binarize = _Binarize(
            onset=self.vad_onset,
            offset=self.vad_offset,
            min_duration_on=0.1,
            min_duration_off=0.1,
            max_duration=30.0,
        )
        segments = binarize(scores, frame_times)
        return [(float(s.start), float(s.end)) for s in segments]

    # Diarization pipeline; split hurts flow.
    def __call__(  # noqa: PLR0912, PLR0915
        self,
        audio: str | np.ndarray,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        return_embeddings: bool = False,
        progress_callback: ProgressCallback = None,
    ) -> tuple[pd.DataFrame, dict[str, list[float]] | None] | pd.DataFrame:
        """Perform speaker diarization on audio.

        Returns diarization dataframe, optionally with speaker embeddings.
        """
        # Lazy import: mlx.core loads the MLX runtime only at call time.
        import mlx.core as mx  # noqa: PLC0415  # pyrefly: ignore[missing-import]

        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_mx = mx.array(audio, dtype=mx.float32)

        # 1. Segmentation -> speech regions.
        scores, frame_times = self._segment_audio(audio_mx)
        scores_np = np.array(scores)
        frame_times_np = np.array(frame_times)
        speech_segments = self._binarize_segments(scores_np, frame_times_np)

        if not speech_segments:
            logger.warning("No speech segments found for diarization.")
            empty_df = pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])
            return (empty_df, None) if return_embeddings else empty_df

        if progress_callback is not None:
            progress_callback(50.0)

        # 2. Extract WeSpeaker embedding per speech sub-segment.
        # Split long segments into 3s windows with 1s step so we get
        # multiple embeddings per speaker for clustering.
        embeddings = []
        seg_info = []
        for start, end in speech_segments:
            dur = end - start
            if dur < EMB_WINDOW:
                f1 = max(0, int(start * SAMPLE_RATE))
                f2 = min(len(audio), int(end * SAMPLE_RATE))
                if f2 - f1 < SAMPLE_RATE * 0.3:
                    continue
                seg_audio = audio_mx[f1:f2]
                emb = self._embed(seg_audio, weights=self._wespeaker_weights)
                embeddings.append(np.array(emb))
                seg_info.append((start, end))
                continue
            n_windows = int((dur - EMB_WINDOW) / EMB_STEP) + 1
            for w in range(n_windows):
                w_start = start + w * EMB_STEP
                w_end = w_start + EMB_WINDOW
                w_end = min(w_end, end)
                if w_end - w_start < 0.5:
                    break
                f1 = int(w_start * SAMPLE_RATE)
                f2 = int(w_end * SAMPLE_RATE)
                seg_audio = audio_mx[f1:f2]
                emb = self._embed(seg_audio, weights=self._wespeaker_weights)
                embeddings.append(np.array(emb))
                seg_info.append((w_start, w_end))

        if not embeddings:
            logger.warning("No segments long enough for embedding extraction.")
            empty_df = pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])
            return (empty_df, None) if return_embeddings else empty_df

        embeddings = np.array(embeddings)

        if progress_callback is not None:
            progress_callback(80.0)

        # 3. Cluster embeddings.
        n = len(embeddings)
        if num_speakers is not None:
            n_clusters = min(num_speakers, n)
        elif min_speakers is not None and max_speakers is not None:
            n_clusters = self._estimate_clusters(embeddings, min_speakers, max_speakers)
        elif max_speakers is not None:
            n_clusters = self._estimate_clusters(embeddings, 1, max_speakers)
        else:
            n_clusters = self._estimate_clusters(embeddings, 1, min(n, 8))

        n_clusters = max(1, min(n_clusters, n))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        if progress_callback is not None:
            progress_callback(100.0)

        # 4. Build dataframe.
        rows = []
        for i, (start, end) in enumerate(seg_info):
            speaker = f"SPEAKER_{labels[i]:02d}"
            seg_obj = Segment(start, end, speaker)
            rows.append(
                {
                    "segment": seg_obj,
                    "label": labels[i],
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                }
            )
        diarize_df = pd.DataFrame(rows)

        if return_embeddings:
            speaker_embeddings = {}
            for spk_label in sorted(set(labels)):
                mask = labels == spk_label
                mean_emb = embeddings[mask].mean(axis=0)
                speaker_embeddings[f"SPEAKER_{spk_label:02d}"] = mean_emb.tolist()
            return diarize_df, speaker_embeddings

        return diarize_df

    @staticmethod
    def _estimate_clusters(embeddings, min_s, max_s):
        """Pick cluster count via cosine-distance gap heuristic."""
        # Lazy imports: scipy is heavy and only needed for clustering.
        from scipy.cluster.hierarchy import fcluster, linkage  # noqa: PLC0415
        from scipy.spatial.distance import pdist  # noqa: PLC0415

        dists = pdist(embeddings, metric="cosine")
        if len(dists) == 0:
            return min_s
        # Linkage matrix; uppercase mirrors scipy's conventional name.
        Z = linkage(dists, method="average")  # noqa: N806
        best_n = min_s
        best_score = -1.0
        for n in range(min_s, min(max_s, len(embeddings)) + 1):
            labels = fcluster(Z, t=n, criterion="maxclust")
            # Silhouette-like: mean intra-cluster cosine sim.
            sims = []
            for c in set(labels):
                mask = labels == c
                if mask.sum() < 2:
                    continue
                cluster_embs = embeddings[mask]
                sim = cluster_embs @ cluster_embs.T
                sims.append(sim.mean())
            score = np.mean(sims) if sims else 0.0
            if score > best_score:
                best_score = score
                best_n = n
        return best_n


def assign_word_speakers(  # noqa: PLR0912
    diarize_df: pd.DataFrame,
    transcript_result: AlignedTranscriptionResult | TranscriptionResult,
    speaker_embeddings: dict[str, list[float]] | None = None,
    fill_nearest: bool = False,
) -> AlignedTranscriptionResult | TranscriptionResult:
    """Assign speakers to words and segments in the transcript.

    Uses an interval tree for O(log n) overlap queries.
    """
    transcript_segments = transcript_result.get("segments", [])
    if not transcript_segments or diarize_df is None or len(diarize_df) == 0:
        return transcript_result

    intervals = [(row["start"], row["end"], row["speaker"]) for _, row in diarize_df.iterrows()]
    tree = IntervalTree(intervals)

    for seg in transcript_segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)

        overlaps = tree.query(seg_start, seg_end)

        if overlaps:
            speaker_intersections: dict[str, float] = {}
            for speaker, intersection in overlaps:
                speaker_intersections[speaker] = (
                    speaker_intersections.get(speaker, 0.0) + intersection
                )
            seg["speaker"] = max(speaker_intersections.items(), key=lambda x: x[1])[0]
        elif fill_nearest:
            seg_mid = (seg_start + seg_end) / 2
            nearest_speaker = tree.find_nearest(seg_mid)
            if nearest_speaker:
                seg["speaker"] = nearest_speaker

        if "words" in seg:
            # 'words' only exists on aligned segments; narrowed by the in-check.
            words: list[SingleWordSegment] = seg.get("words") or []  # pyrefly: ignore[bad-assignment]
            for word in words:
                if "start" not in word:
                    continue
                word_start = word["start"]
                word_end = word.get("end", word_start)
                word_overlaps = tree.query(word_start, word_end)
                if word_overlaps:
                    speaker_intersections = {}
                    for speaker, intersection in word_overlaps:
                        speaker_intersections[speaker] = (
                            speaker_intersections.get(speaker, 0.0) + intersection
                        )
                    word["speaker"] = max(speaker_intersections.items(), key=lambda x: x[1])[0]
                elif fill_nearest:
                    word_mid = (word_start + word_end) / 2
                    nearest_speaker = tree.find_nearest(word_mid)
                    if nearest_speaker:
                        word["speaker"] = nearest_speaker

    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings

    return transcript_result


class Segment:
    def __init__(self, start: float, end: float, speaker: str | None = None):
        self.start = start
        self.end = end
        self.speaker = speaker
