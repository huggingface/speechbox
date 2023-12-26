import numpy as np


def IoU(diarized_segments: np.ndarray, asr_segments: np.ndarray) -> np.ndarray:
    """
    Calculates the Intersection over Union (IoU) between diarized_segments and asr_segments.

    Args:
    -----------
    - diarized_segments (np.ndarray): An array representing N segments with shape (M, 2), where each row
      contains the start and end times of a diarized segment.
    - asr_segments (np.ndarray): An array representing M segments with shape (N, 2), where each row contains
      the start and end times of an asr segment.

    Returns:
    --------
    - np.ndarray: A 2D array of shape (N, M) representing the IoU between each pair of diarized and.
      The value at position (i, j) in the array corresponds to the IoU between the asr segment i and the diarized segment j.
      Values are in the range [0, 1], where 0 indicates no intersection and 1 indicates perfect overlap.

    Note:
    - The IoU is calculated as the ratio of the intersection over the union of the time intervals.
    - Segments with no overlap result in an IoU value of 0.
    - Segments with overlap but no intersection (e.g., one segment completely contained within another) can
      have an IoU greater than 0.

    Example:
    ```python
        diarized_segments = np.array([[0, 5], [3, 8], [6, 10]])
        asr_segments = np.array([[2, 6], [1, 4]])

        IoU_values = IoU(diarized_segments, asr_segments)
        print(IoU_values)
        # Output
        # [[0.5        0.5        0.]
        # [0.6         0.14285714 0.]]
    ```
    """
    # We measure intersection between each of the N asr_segments [Nx2] and each M of diarize_ segments [Mx2]
    # The result is a NxM matrix. intersection <= 0 mean no intersection.
    starts = np.maximum(asr_segments[:, 0, np.newaxis], diarized_segments[:, 0])
    ends = np.minimum(asr_segments[:, 1, np.newaxis], diarized_segments[:, 1])
    intersections = np.maximum(ends - starts, 0)

    # Union for segments without overlap will lead to invalid results but it does not matters
    # as we opt them out eventually.
    union = np.maximum(asr_segments[:, 1, np.newaxis], diarized_segments[:, 1]) - np.minimum(
        asr_segments[:, 0, np.newaxis], diarized_segments[:, 0]
    )

    # Negative results are zeroed as they are invalid.
    intersection_over_union = np.maximum(intersections / union, 0)

    return intersection_over_union


def match_segments(
    diarized_segments: np.ndarray,
    diarized_labels: list[str],
    asr_segments: np.ndarray,
    threshold: float = 0.0,
    no_match_label: str = "NO_SPEAKER",
) -> np.ndarray:
    """
    Perform segment matching between diarized segments and ASR (Automatic Speech Recognition) segments.

    Args:
    -----
    - diarized_segments (np.ndarray): Array representing diarized speaker segments.
    - diarized_labels (list[str]): List of labels corresponding to diarized_segments.
    - asr_segments (np.ndarray): Array representing ASR speaker segments.
    - threshold (float, optional): IoU (Intersection over Union) threshold for matching. Default is 0.0.
    - no_match_label (str, optional): Label assigned when no matching segment is found. Default is "NO_SPEAKER".

    Returns:
    --------
    - np.ndarray: Array of labels corresponding to the best-matched ASR segments for each diarized segment.

    Notes:
    - The function calculates IoU between diarized segments and ASR segments and considers only segments with IoU above the threshold.
    - If no matching segment is found, the specified `no_match_label` is assigned.
    - The returned array represents the labels of the best-matched ASR segments for each diarized segment.
    """
    iou_results = IoU(diarized_segments, asr_segments)
    # Zero out iou below threshold.
    iou_results[iou_results <= threshold] = 0.0
    # We create a no match label which value will be threshold
    diarized_labels = [no_match_label] + diarized_labels
    # If there is nothing above threshold, no_match_label will be assigned.
    iou_results = np.hstack([threshold * np.ones((iou_results.shape[0], 1)), iou_results])
    # Will find argument with highest iou (if all zeroes, will assign first (no_match_label)).
    best_match_idx = np.argmax(iou_results, axis=1)
    assigned_labels = np.take(diarized_labels, best_match_idx)

    return assigned_labels
