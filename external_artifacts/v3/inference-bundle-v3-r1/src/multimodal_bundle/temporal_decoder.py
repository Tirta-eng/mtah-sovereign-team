"""Viterbi HMM temporal decoding for Multimodal Bundle V3."""

from typing import Dict, List, Sequence, Tuple
import numpy as np

from .contracts import FrameKey, LABELS

STEP_US = 40_000  # 25 fps step

def contiguous_runs(keys: Sequence[FrameKey], stamps: Sequence[int], valid: Sequence[bool]) -> List[Tuple[int, int]]:
    """Return [start, stop) index spans with valid targets and adjacent 25 Hz keys."""
    if not (len(keys) == len(stamps) == len(valid)):
        raise ValueError("Timeline length mismatch")
    runs, start = [], None
    for i, (key, stamp, good) in enumerate(zip(keys, stamps, valid)):
        adjacent = (
            i > 0
            and (key.match_id, key.period, key.source_frame - 1, stamp - STEP_US)
            == (keys[i - 1].match_id, keys[i - 1].period, keys[i - 1].source_frame, stamps[i - 1])
        )
        if start is not None and (not good or not adjacent):
            runs.append((start, i))
            start = None
        if good and start is None:
            start = i
    if start is not None:
        runs.append((start, len(keys)))
    return runs


def _check_prob_array(values: np.ndarray, shape: Tuple[int, ...], name: str, positive: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != shape or not np.isfinite(arr).all() or (arr < 0).any() or (arr > 1).any():
        raise ValueError(f"Invalid probability array {name}")
    if positive and not (arr > 0).all():
        raise ValueError(f"{name} must be strictly positive")
    if not np.allclose(arr.sum(axis=-1), 1.0, atol=1e-6, rtol=0):
        raise ValueError(f"{name} must sum to 1.0 along last axis")
    return arr


def decode_probabilities(
    keys: Sequence[FrameKey],
    stamps: Sequence[int],
    probabilities: np.ndarray,
    parameters: Dict[str, object],
) -> np.ndarray:
    """Perform offline Viterbi HMM decoding across contiguous temporal runs."""
    if len(keys) != len(stamps):
        raise ValueError("Prediction timeline length mismatch")
    if len(keys) != len(probabilities):
        raise ValueError("Probability length mismatch with keys")

    probs = _check_prob_array(probabilities, (len(keys), 3), "posteriors")
    prior = _check_prob_array(np.array(parameters["state_prior"]), (3,), "state prior", positive=True)
    initial = _check_prob_array(np.array(parameters["initial_prior"]), (3,), "initial prior", positive=True)
    matrix = _check_prob_array(np.array(parameters["transition_matrix"]), (3, 3), "transition matrix", positive=True)

    log_posterior = np.full_like(probs, -np.inf)
    np.log(probs, out=log_posterior, where=probs > 0)
    emissions = log_posterior - np.log(prior)
    log_matrix = np.log(matrix)

    result = np.empty(len(keys), dtype="U1")

    for start, stop in contiguous_runs(keys, stamps, [True] * len(keys)):
        n_steps = stop - start
        parents = np.zeros((n_steps, 3), dtype=np.int8)
        score = np.log(initial) + emissions[start]

        for offset in range(1, n_steps):
            candidates = score[:, None] + log_matrix
            parents[offset] = np.argmax(candidates, axis=0)
            score = candidates[parents[offset], np.arange(3)] + emissions[start + offset]
            score -= score.max()  # Prevent numerical overflow

        state = int(np.argmax(score))
        for offset in range(n_steps - 1, -1, -1):
            result[start + offset] = LABELS[state]
            state = int(parents[offset, state])

    return result
